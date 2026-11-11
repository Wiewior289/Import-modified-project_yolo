package com.tencent.yolo11ncnn

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.util.Log
import android.util.Size
import androidx.appcompat.app.AppCompatActivity
import android.content.Context
import android.graphics.ImageFormat
import android.hardware.HardwareBuffer
import android.hardware.camera2.*
import android.media.ImageReader
import android.os.Handler
import android.os.HandlerThread
import android.os.SystemClock
import android.view.View
import android.widget.AdapterView
import android.widget.ArrayAdapter
import android.widget.LinearLayout
import android.widget.Spinner
import android.widget.SeekBar
import android.widget.TextView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicIntegerArray

/**
 * ============================================================================
 * MainActivity - Camera2 / AHardwareBuffer YOLO test harness
 * Platform: Samsung Galaxy S9+ (Mali-G72 MP18 + Exynos 9810)
 *
 * CPU / Vulkan baseline: AHardwareBuffer YUV planes are mapped and converted on CPU.
 * Vulkan + AHB: Camera2 buffer is imported directly into Vulkan.
 * All modes return output to CPU for DFL decode, NMS and dashboard telemetry.
 * ============================================================================
 */
class MainActivity : AppCompatActivity() {

    private val yoloEngine = Yolo11Ncnn()
    @Volatile private var selectedComputeMode: Int = Yolo11Ncnn.COMPUTE_VULKAN_AHB
    @Volatile private var modeGeneration: Long = 0L
    private var modeSelectorInitialized = false
    private var modelSelectorInitialized = false
    @Volatile private var selectedModelType: Int = Yolo11Ncnn.MODEL_YOLO11N
    @Volatile private var selectedTargetSize: Int = TARGET_SIZE_MIN
    @Volatile private var asyncDrainPending: Boolean = false
    private var selectableModelTypes: List<Int> = listOf(Yolo11Ncnn.MODEL_YOLO11N)
    private lateinit var telemetryServer: TelemetryServer
    private lateinit var backgroundInitExecutor: ExecutorService
    private lateinit var gpuFrameExecutor: ExecutorService
    private lateinit var cpuFrameExecutor: ExecutorService

    // Camera2 background thread & components
    private var cameraThread: HandlerThread? = null
    private var cameraHandler: Handler? = null
    private var cameraDevice: CameraDevice? = null
    private var captureSession: CameraCaptureSession? = null
    private var imageReader: ImageReader? = null
    private var sensorOrientation: Int = 90

    // Pre-allocated flat buffer: 1 count float + 64 detections * 7 floats (449 floats)
    // ZERO HEAP ALLOCATIONS ON THE FRAME LOOP!
    private val detectionBuffer = FloatArray(1 + 64 * 7)
    private val asyncDetectionBuffers = Array(2) { FloatArray(1 + 64 * 7) }
    private val asyncSlotBusy = AtomicIntegerArray(2)

    @Volatile private var confThreshold: Float = 0.25f
    @Volatile private var nmsThreshold: Float = 0.45f

    private val isYoloReady = AtomicBoolean(false)
    private val isCameraRunning = AtomicBoolean(false)
    private val isProcessingFrame = AtomicBoolean(false)

    private var frameSequence: Long = 0L
    private var fpsWindowStartMs: Long = 0L
    private var fpsWindowFrames: Int = 0
    private var currentFps: Float = 0f

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        backgroundInitExecutor = Executors.newSingleThreadExecutor()
        gpuFrameExecutor = Executors.newSingleThreadExecutor()
        cpuFrameExecutor = Executors.newSingleThreadExecutor()

        // Start telemetry before installing the selector, so an early spinner
        // callback can safely report a mode change.
        startTelemetryServer()
        installComputeModeSelector()

        // KROK 2 & 3: Weryfikacja uprawnień i asynchroniczny start potoku
        if (allPermissionsGranted()) {
            launchAsyncPipeline()
        } else {
            telemetryServer.broadcastStatus("AWAITING_PERMISSIONS", "Oczekiwanie na akceptację uprawnień...")
            ActivityCompat.requestPermissions(this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS)
        }
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQUEST_CODE_PERMISSIONS) {
            if (allPermissionsGranted()) {
                Log.i(TAG, "Uprawnienia przyznane, uruchamianie potoku")
                launchAsyncPipeline()
            } else {
                Log.e(TAG, "Brak uprawnień kamery!")
                telemetryServer.broadcastStatus("ERROR_PERMISSIONS", "Brak uprawnień do kamery!")
            }
        }
    }

    private fun installComputeModeSelector() {
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(20, 16, 20, 8)
        }

        val computeLabel = TextView(this).apply {
            text = "Tryb obliczeniowy NCNN"
            textSize = 16f
        }
        val computeSpinner = Spinner(this)
        val computeLabels = listOf(
            "CPU",
            "GPU Vulkan",
            "GPU Vulkan + AHB",
            "GPU AHB prefix + CPU suffix (Point B)",
            "GPU AHB + CPU suffix (N/N+1)",
            "DIAG: AHB vs CPU YUV + Vulkan (10 samples)",
            "DIAG: MatMul Vulkan vs CPU (same frame, 10 samples)"
        )
        val computeAdapter = ArrayAdapter(this, android.R.layout.simple_spinner_item, computeLabels)
        computeAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
        computeSpinner.adapter = computeAdapter
        computeSpinner.setSelection(selectedComputeMode)
        computeSpinner.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(parent: AdapterView<*>?, view: View?, position: Int, id: Long) {
                if (!modeSelectorInitialized) {
                    modeSelectorInitialized = true
                    if (position != selectedComputeMode) switchComputeMode(position)
                    return
                }
                switchComputeMode(position)
            }

            override fun onNothingSelected(parent: AdapterView<*>?) {}
        }

        val modelLabelView = TextView(this).apply {
            text = "Model detekcji"
            textSize = 16f
            setPadding(0, 12, 0, 0)
        }

        val assetNames = assets.list("")?.toSet() ?: emptySet()
        val modelTypes = mutableListOf(Yolo11Ncnn.MODEL_YOLO11N)
        val modelLabels = mutableListOf("YOLO11n (Nano)")
        if (assetNames.contains("yolo11s.param") && assetNames.contains("yolo11s.bin")) {
            modelTypes.add(Yolo11Ncnn.MODEL_YOLO11S)
            modelLabels.add("YOLO11s (Small)")
        } else {
            Log.w(TAG, "YOLO11s selector disabled: both yolo11s.param and yolo11s.bin must exist in app/src/main/assets")
        }
        selectableModelTypes = modelTypes

        val modelSpinner = Spinner(this)
        val modelAdapter = ArrayAdapter(this, android.R.layout.simple_spinner_item, modelLabels)
        modelAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
        modelSpinner.adapter = modelAdapter
        val initialModelPosition = selectableModelTypes.indexOf(selectedModelType).coerceAtLeast(0)
        modelSpinner.setSelection(initialModelPosition)
        modelSpinner.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(parent: AdapterView<*>?, view: View?, position: Int, id: Long) {
                val modelType = selectableModelTypes.getOrElse(position) { Yolo11Ncnn.MODEL_YOLO11N }
                if (!modelSelectorInitialized) {
                    modelSelectorInitialized = true
                    if (modelType != selectedModelType) switchModelType(modelType)
                    return
                }
                switchModelType(modelType)
            }

            override fun onNothingSelected(parent: AdapterView<*>?) {}
        }

        val resolutionLabel = TextView(this).apply {
            text = "Rozdzielczość wejściowa modelu"
            textSize = 16f
            setPadding(0, 12, 0, 0)
        }
        val resolutionValue = TextView(this).apply {
            text = selectedTargetSize.toString() + " x " + selectedTargetSize
            textSize = 14f
        }
        val resolutionSeekBar = SeekBar(this).apply {
            max = (TARGET_SIZE_MAX - TARGET_SIZE_MIN) / TARGET_SIZE_STEP
            progress = (selectedTargetSize - TARGET_SIZE_MIN) / TARGET_SIZE_STEP
        }
        resolutionSeekBar.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(seekBar: SeekBar?, progress: Int, fromUser: Boolean) {
                val previewSize = TARGET_SIZE_MIN + progress * TARGET_SIZE_STEP
                resolutionValue.text = previewSize.toString() + " x " + previewSize
            }

            override fun onStartTrackingTouch(seekBar: SeekBar?) {}

            override fun onStopTrackingTouch(seekBar: SeekBar?) {
                val progress = seekBar?.progress ?: return
                val newSize = TARGET_SIZE_MIN + progress * TARGET_SIZE_STEP
                switchTargetSize(newSize)
            }
        })

        root.addView(computeLabel, LinearLayout.LayoutParams(
            LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT
        ))
        root.addView(computeSpinner, LinearLayout.LayoutParams(
            LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT
        ))
        root.addView(modelLabelView, LinearLayout.LayoutParams(
            LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT
        ))
        root.addView(modelSpinner, LinearLayout.LayoutParams(
            LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT
        ))
        root.addView(resolutionLabel, LinearLayout.LayoutParams(
            LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT
        ))
        root.addView(resolutionValue, LinearLayout.LayoutParams(
            LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT
        ))
        root.addView(resolutionSeekBar, LinearLayout.LayoutParams(
            LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT
        ))
        setContentView(root)
    }

    private fun computeModeLabel(mode: Int): String = when (mode) {
        Yolo11Ncnn.COMPUTE_CPU -> "CPU"
        Yolo11Ncnn.COMPUTE_VULKAN -> "GPU Vulkan"
        Yolo11Ncnn.COMPUTE_VULKAN_AHB -> "GPU Vulkan + AHB"
        Yolo11Ncnn.COMPUTE_VULKAN_AHB_SPLIT -> "GPU AHB prefix + CPU suffix (Point B)"
        Yolo11Ncnn.COMPUTE_VULKAN_AHB_ASYNC -> "GPU AHB + CPU suffix (N/N+1)"
        Yolo11Ncnn.COMPUTE_AHB_COMPARE -> "DIAG: AHB vs CPU YUV + Vulkan"
        Yolo11Ncnn.COMPUTE_MATMUL_VULKAN_COMPARE -> "DIAG: MatMul Vulkan vs CPU (same frame)"
        else -> "UNKNOWN($mode)"
    }

    private fun modelLabel(modelType: Int): String = when (modelType) {
        Yolo11Ncnn.MODEL_YOLO11N -> "YOLO11n"
        Yolo11Ncnn.MODEL_YOLO11S -> "YOLO11s"
        Yolo11Ncnn.MODEL_YOLO11M -> "YOLO11m"
        else -> "UNKNOWN_MODEL(" + modelType + ")"
    }

    private fun switchModelType(newModelType: Int) {
        if (!selectableModelTypes.contains(newModelType) || newModelType == selectedModelType) return
        val previousMode = selectedComputeMode
        selectedModelType = newModelType
        reinitializePipeline("MODEL_SWITCH", previousMode)
    }

    private fun switchTargetSize(newSize: Int) {
        if (newSize < TARGET_SIZE_MIN || newSize > TARGET_SIZE_MAX ||
            (newSize - TARGET_SIZE_MIN) % TARGET_SIZE_STEP != 0 ||
            newSize == selectedTargetSize) return
        val previousMode = selectedComputeMode
        selectedTargetSize = newSize
        reinitializePipeline("INPUT_SIZE_SWITCH", previousMode)
    }

    private fun reinitializePipeline(reason: String, previousMode: Int) {
        val generation = ++modeGeneration
        val modelAtRequest = selectedModelType
        val targetAtRequest = selectedTargetSize
        val computeModeAtRequest = selectedComputeMode
        val mustDrainAsync = previousMode == Yolo11Ncnn.COMPUTE_VULKAN_AHB_ASYNC || asyncDrainPending
        if (mustDrainAsync) asyncDrainPending = true

        isYoloReady.set(false)
        isCameraRunning.set(false)
        closeCameraForModeSwitch()

        val configLabel = modelLabel(modelAtRequest) + " / " +
            targetAtRequest + "x" + targetAtRequest + " / " + computeModeLabel(computeModeAtRequest)
        telemetryServer.broadcastStatus("COMPUTE_MODE_SWITCH", reason + ": " + configLabel)

        val enqueueReinitialization = {
            backgroundInitExecutor.execute {
                if (generation != modeGeneration) return@execute
                fpsWindowStartMs = 0L
                fpsWindowFrames = 0
                currentFps = 0f
                asyncSlotBusy.set(0, 0)
                asyncSlotBusy.set(1, 0)
                yoloEngine.Destroy()

                val startedAt = SystemClock.elapsedRealtime()
                val initialized = try {
                    yoloEngine.Init(assets, modelAtRequest, targetAtRequest, computeModeAtRequest)
                } catch (e: Exception) {
                    Log.e(TAG, "Błąd inicjalizacji " + configLabel, e)
                    false
                }
                val durationMs = SystemClock.elapsedRealtime() - startedAt
                if (generation == modeGeneration) asyncDrainPending = false

                runOnUiThread {
                    if (generation != modeGeneration) return@runOnUiThread
                    imageReader?.setOnImageAvailableListener(null, null)
                    imageReader?.close()
                    imageReader = null
                    if (initialized) {
                        isYoloReady.set(true)
                        Log.i(TAG, configLabel + " gotowy w " + durationMs + "ms")
                        telemetryServer.broadcastStatus(
                            "COMPUTE_MODE_READY",
                            configLabel + " gotowy (" + durationMs + "ms)"
                        )
                        startCamera2()
                    } else {
                        isYoloReady.set(false)
                        telemetryServer.broadcastStatus(
                            "ERROR_COMPUTE_MODE",
                            "Nie udało się uruchomić " + configLabel
                        )
                    }
                }
            }
        }

        if (mustDrainAsync) {
            val detachAndDrain = {
                imageReader?.setOnImageAvailableListener(null, null)
                gpuFrameExecutor.execute {
                    cpuFrameExecutor.execute { enqueueReinitialization() }
                }
            }
            val handler = cameraHandler
            val posted = handler?.post { detachAndDrain() } ?: false
            if (!posted) detachAndDrain()
        } else {
            enqueueReinitialization()
        }
    }

    private fun switchComputeMode(newMode: Int) {
        val previousMode = selectedComputeMode
        if ((newMode !in Yolo11Ncnn.COMPUTE_CPU..Yolo11Ncnn.COMPUTE_MATMUL_VULKAN_COMPARE) ||
            newMode == previousMode) return
        selectedComputeMode = newMode
        reinitializePipeline("COMPUTE_MODE_SWITCH", previousMode)
    }

    private fun closeCameraForModeSwitch() {
        try {
            imageReader?.setOnImageAvailableListener(null, null)
            captureSession?.close()
            captureSession = null
            cameraDevice?.close()
            cameraDevice = null
        } catch (e: Exception) {
            Log.e(TAG, "Błąd zatrzymywania kamery przy zmianie backendu", e)
        }
    }

    private fun startTelemetryServer() {
        try {
            telemetryServer = TelemetryServer(
                port = 8080,
                onCommandReceived = { cmd, payload ->
                    handleRemoteCommand(cmd, payload)
                }
            )
            telemetryServer.start()
            Log.i(TAG, "TelemetryServer wystartował na porcie 8080")
            telemetryServer.broadcastStatus("SERVER_ONLINE", "Serwer WebSocket nasłuchuje na porcie 8080")
        } catch (e: Exception) {
            Log.e(TAG, "Błąd startu TelemetryServer", e)
        }
    }

    private fun launchAsyncPipeline() {
        val modeAtStart = selectedComputeMode
        val modelAtStart = selectedModelType
        val targetSizeAtStart = selectedTargetSize
        val generationAtStart = modeGeneration
        backgroundInitExecutor.execute {
            Log.i(TAG, "Ładowanie " + modelLabel(modelAtStart) + " input=" +
                targetSizeAtStart + "x" + targetSizeAtStart + " backend=" +
                computeModeLabel(modeAtStart) + " w tle...")
            telemetryServer.broadcastStatus(
                "INIT_YOLO",
                "Inicjalizacja YOLO11: " + computeModeLabel(modeAtStart)
            )

            val startTime = SystemClock.elapsedRealtime()
            val initSuccess = try {
                yoloEngine.Init(assets, modelAtStart, targetSizeAtStart, modeAtStart)
            } catch (e: Exception) {
                Log.e(TAG, "Błąd podczas init backendu " + computeModeLabel(modeAtStart), e)
                false
            }
            val loadDuration = SystemClock.elapsedRealtime() - startTime

            runOnUiThread {
                if (generationAtStart != modeGeneration) return@runOnUiThread
                if (initSuccess) {
                    Log.i(TAG, modelLabel(modelAtStart) + " input=" + targetSizeAtStart +
                        " backend=" + computeModeLabel(modeAtStart) + " Init sukces w " + loadDuration + "ms")
                    isYoloReady.set(true)
                    telemetryServer.broadcastStatus(
                        "YOLO_READY",
                        modelLabel(modelAtStart) + " " + targetSizeAtStart + "x" +
                            targetSizeAtStart + " / " + computeModeLabel(modeAtStart) +
                            " gotowy (" + loadDuration + "ms)"
                    )
                    startCamera2()
                } else {
                    isYoloReady.set(false)
                    Log.e(TAG, "Nie udało się zainicjalizować backendu " + computeModeLabel(modeAtStart))
                    telemetryServer.broadcastStatus(
                        "ERROR_YOLO_INIT",
                        "Init nie powiódł się: " + computeModeLabel(modeAtStart)
                    )
                }
            }
        }
    }

    private fun handleRemoteCommand(cmd: String, payload: String) {
        if (cmd != "SET_PARAMS") return
        try {
            val q = '"'
            val confKey = "" + q + "conf" + q
            val nmsKey = "" + q + "nms" + q
            val conf = payload.substringAfter(confKey, "").substringAfter(":", "")
                .substringBefore(",").substringBefore("}").replace("\"", "").trim().toFloatOrNull()
            val nms = payload.substringAfter(nmsKey, "").substringAfter(":", "")
                .substringBefore(",").substringBefore("}").replace("\"", "").trim().toFloatOrNull()
            if (conf != null) confThreshold = conf.coerceIn(0.01f, 0.99f)
            if (nms != null) nmsThreshold = nms.coerceIn(0.01f, 0.99f)
            telemetryServer.broadcastStatus("PARAMS_UPDATED", "conf=" + confThreshold + ", nms=" + nmsThreshold)
        } catch (e: Exception) {
            Log.e(TAG, "Błąd parsowania SET_PARAMS", e)
        }
    }

    private fun reserveAsyncSlot(): Int {
        for (slot in 0..1) {
            if (asyncSlotBusy.compareAndSet(slot, 0, 1)) return slot
        }
        return -1
    }

    private fun queueSplitFrameSync(
        image: android.media.Image,
        frameId: Long,
        imageTimestampNs: Long,
        rotateType: Int
    ) {
        backgroundInitExecutor.execute {
            try {
                var prefixRet = -1
                var hwBuffer: HardwareBuffer? = null
                try {
                    val acquiredBuffer = image.hardwareBuffer
                    hwBuffer = acquiredBuffer
                    prefixRet = if (acquiredBuffer == null) {
                        -70
                    } else {
                        yoloEngine.PrefixFrame(acquiredBuffer, 0, rotateType, frameId, imageTimestampNs)
                    }
                } catch (e: Exception) {
                    Log.e(TAG, "Point B GPU prefix exception frame=$frameId", e)
                    prefixRet = -71
                } finally {
                    try {
                        hwBuffer?.close()
                    } catch (e: Exception) {
                        Log.e(TAG, "HardwareBuffer wrapper close failed frame=$frameId", e)
                    }
                    try {
                        image.close()
                    } catch (e: Exception) {
                        Log.e(TAG, "Camera Image close failed frame=$frameId", e)
                    }
                }

                if (prefixRet != 0) {
                    Log.e(TAG, "Point B GPU prefix failed frame=$frameId ret=$prefixRet")
                    return@execute
                }

                detectionBuffer.fill(0f)
                val suffixRet = yoloEngine.SuffixFrame(0, detectionBuffer, confThreshold, nmsThreshold)
                updateProcessingFps()
                val count = suffixRet.coerceAtLeast(0)
                val topScore = if (suffixRet >= 0) yoloEngine.GetTopScore() else 0f
                val topClass = if (suffixRet >= 0) yoloEngine.GetTopClass() else -1

                if (frameId % 30L == 0L) {
                    Log.i(TAG, "POINT_B_FRAME frame=$frameId imageTs=$imageTimestampNs " +
                        "FPS=" + String.format(java.util.Locale.US, "%.1f", currentFps) +
                        " detections=$count top=" +
                        String.format(java.util.Locale.US, "%.3f", topScore) +
                        " class=$topClass")
                }
                if (suffixRet < 0) {
                    Log.e(TAG, "Point B CPU suffix failed frame=$frameId ret=$suffixRet")
                }
                val jsonPayload = buildJson(
                    count, detectionBuffer, currentFps, topScore, topClass, frameId, imageTimestampNs
                )
                try {
                    telemetryServer.sendDetections(jsonPayload)
                } catch (e: Exception) {
                    Log.e(TAG, "Błąd wysyłania telemetrii Point B", e)
                }
                sendToSTM32(count, detectionBuffer, false)
            } catch (e: Exception) {
                Log.e(TAG, "Point B frame exception frame=$frameId", e)
            } finally {
                isProcessingFrame.set(false)
            }
        }
    }

    private fun queueAsyncFrame(image: android.media.Image, frameId: Long, imageTimestampNs: Long, rotateType: Int) {
        val slot = reserveAsyncSlot()
        if (slot < 0) {
            image.close()
            return
        }

        try {
            gpuFrameExecutor.execute {
                var prefixRet = -1
                var hwBuffer: HardwareBuffer? = null
                try {
                    val acquiredBuffer = image.hardwareBuffer
                    hwBuffer = acquiredBuffer
                    if (acquiredBuffer == null) {
                        prefixRet = -70
                    } else {
                        prefixRet = yoloEngine.PrefixFrame(
                            acquiredBuffer, slot, rotateType, frameId, imageTimestampNs
                        )
                    }
                } catch (e: Exception) {
                    Log.e(TAG, "GPU prefix exception frame=$frameId slot=$slot", e)
                    prefixRet = -71
                } finally {
                    try {
                        hwBuffer?.close()
                    } catch (e: Exception) {
                        Log.e(TAG, "HardwareBuffer wrapper close failed frame=$frameId", e)
                    }
                    try {
                        image.close()
                    } catch (e: Exception) {
                        Log.e(TAG, "Camera Image close failed frame=$frameId", e)
                    }
                }

                if (prefixRet != 0) {
                    Log.e(TAG, "GPU prefix failed frame=$frameId slot=$slot ret=$prefixRet")
                    asyncSlotBusy.set(slot, 0)
                    return@execute
                }

                try {
                    cpuFrameExecutor.execute {
                        val output = asyncDetectionBuffers[slot]
                        output.fill(0f)
                        var suffixRet = -1
                        try {
                            suffixRet = yoloEngine.SuffixFrame(
                                slot, output, confThreshold, nmsThreshold
                            )
                            updateProcessingFps()
                            val count = suffixRet.coerceAtLeast(0)
                            val topScore = if (suffixRet >= 0) yoloEngine.GetTopScore() else 0f
                            val topClass = if (suffixRet >= 0) yoloEngine.GetTopClass() else -1

                            if (frameId % 30L == 0L) {
                                Log.i(TAG, "ASYNC_FRAME frame=$frameId slot=$slot imageTs=$imageTimestampNs " +
                                    "FPS=" + String.format(java.util.Locale.US, "%.1f", currentFps) +
                                    " detections=$count top=" +
                                    String.format(java.util.Locale.US, "%.3f", topScore) +
                                    " class=$topClass")
                            }
                            if (suffixRet < 0) {
                                Log.e(TAG, "CPU suffix failed frame=$frameId slot=$slot ret=$suffixRet")
                            }
                            val jsonPayload = buildJson(
                                count, output, currentFps, topScore, topClass, frameId, imageTimestampNs
                            )
                            try {
                                telemetryServer.sendDetections(jsonPayload)
                            } catch (e: Exception) {
                                Log.e(TAG, "Błąd wysyłania telemetrii async", e)
                            }
                            sendToSTM32(count, output, false)
                        } catch (e: Exception) {
                            Log.e(TAG, "CPU suffix exception frame=$frameId slot=$slot", e)
                        } finally {
                            asyncSlotBusy.set(slot, 0)
                        }
                    }
                } catch (e: Exception) {
                    Log.e(TAG, "Nie udało się zakolejkować CPU suffix frame=$frameId", e)
                    asyncSlotBusy.set(slot, 0)
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Nie udało się zakolejkować GPU prefix frame=$frameId", e)
            image.close()
            asyncSlotBusy.set(slot, 0)
        }
    }

    private fun startCamera2() {
        startCameraThread()
        val cameraManager = getSystemService(Context.CAMERA_SERVICE) as CameraManager
        try {
            var selectedCameraId: String? = null
            var selectedCharacteristics: CameraCharacteristics? = null
            for (id in cameraManager.cameraIdList) {
                val characteristics = cameraManager.getCameraCharacteristics(id)
                val facing = characteristics.get(CameraCharacteristics.LENS_FACING)
                if (facing == CameraCharacteristics.LENS_FACING_BACK) {
                    selectedCameraId = id
                    selectedCharacteristics = characteristics
                    sensorOrientation = characteristics.get(CameraCharacteristics.SENSOR_ORIENTATION) ?: 90
                    break
                }
            }

            if (selectedCameraId == null && cameraManager.cameraIdList.isNotEmpty()) {
                selectedCameraId = cameraManager.cameraIdList[0]
                selectedCharacteristics = cameraManager.getCameraCharacteristics(selectedCameraId!!)
                sensorOrientation = selectedCharacteristics?.get(CameraCharacteristics.SENSOR_ORIENTATION) ?: 90
            }

            if (selectedCameraId == null || selectedCharacteristics == null) {
                Log.e(TAG, "Brak dostępnej kamery w urządzeniu!")
                telemetryServer.broadcastStatus("ERROR_CAMERA", "Brak dostępnej kamery!")
                return
            }

            val streamMap = selectedCharacteristics?.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP)
            val yuvSizes = streamMap?.getOutputSizes(ImageFormat.YUV_420_888)?.toList().orEmpty()
            if (yuvSizes.isEmpty()) {
                Log.e(TAG, "Wybrana kamera nie udostępnia strumienia YUV_420_888")
                telemetryServer.broadcastStatus("ERROR_CAMERA_SIZE", "Brak wspieranego wyjścia YUV_420_888")
                return
            }

            // Prefer a square stream; otherwise choose the closest aspect ratio and
            // report the mismatch instead of assuming 320x320 is supported by the HAL.
            val usableSizes = yuvSizes.filter { minOf(it.width, it.height) >= 240 }.ifEmpty { yuvSizes }
            val inputSize = usableSizes.minByOrNull { size ->
                val aspectPenalty = kotlin.math.abs(size.width.toDouble() / size.height.toDouble() - 1.0) * 10000.0
                val areaPenalty = kotlin.math.abs(size.width * size.height - 320 * 320).toDouble() / 1000.0
                aspectPenalty + areaPenalty
            } ?: run {
                telemetryServer.broadcastStatus("ERROR_CAMERA_SIZE", "Nie udało się wybrać rozmiaru YUV")
                return
            }

            val inputAspect = inputSize.width.toDouble() / inputSize.height.toDouble()
            Log.i(TAG, "Camera2 YUV input=" + inputSize.width + "x" + inputSize.height + ", aspect=" + inputAspect)
            if (kotlin.math.abs(inputAspect - 1.0) > 0.02) {
                telemetryServer.broadcastStatus(
                    "WARN_CAMERA_ASPECT",
                    "Strumień kamery nie jest kwadratowy (" + inputSize.width + "x" + inputSize.height + "); preprocess do 320x320 może zmieniać proporcje."
                )
            }

            // Android 10+ ImageReader overload requests GPU sampling and CPU read access.
            imageReader = ImageReader.newInstance(
                inputSize.width, inputSize.height, ImageFormat.YUV_420_888, 3,
                HardwareBuffer.USAGE_CPU_READ_OFTEN or HardwareBuffer.USAGE_GPU_SAMPLED_IMAGE
            )

            imageReader?.setOnImageAvailableListener({ reader ->
                if (!isYoloReady.get() ||
                    (selectedComputeMode != Yolo11Ncnn.COMPUTE_VULKAN_AHB_ASYNC && isProcessingFrame.get())) {
                    reader.acquireLatestImage()?.close()
                    return@setOnImageAvailableListener
                }

                val image = reader.acquireLatestImage() ?: return@setOnImageAvailableListener
                val frameId = ++frameSequence
                val imageTimestampNs = image.timestamp
                if (frameId % 30L == 0L) Log.i(TAG, "CAMERA_IMAGE frame=$frameId imageTs=$imageTimestampNs size=${image.width}x${image.height} crop=${image.cropRect}")
                val rotateType = when (sensorOrientation) {
                    90 -> 6
                    180 -> 3
                    270 -> 8
                    else -> 1
                }

                if (selectedComputeMode == Yolo11Ncnn.COMPUTE_VULKAN_AHB_ASYNC) {
                    queueAsyncFrame(image, frameId, imageTimestampNs, rotateType)
                    return@setOnImageAvailableListener
                }

                if (!isProcessingFrame.compareAndSet(false, true)) {
                    image.close()
                    return@setOnImageAvailableListener
                }

                if (selectedComputeMode == Yolo11Ncnn.COMPUTE_VULKAN_AHB_SPLIT) {
                    queueSplitFrameSync(image, frameId, imageTimestampNs, rotateType)
                    return@setOnImageAvailableListener
                }

                backgroundInitExecutor.execute {
                    try {
                        detectionBuffer.fill(0f)
                        var rawCount = -1
                        val hwBuffer = image.hardwareBuffer
                        if (hwBuffer == null) {
                            // Image.hardwareBuffer is nullable in the Android API.
                            // Keep the frame invalid rather than passing null through JNI.
                            Log.e(TAG, "Image.hardwareBuffer == null for frame=$frameId")
                            rawCount = -4
                        } else {
                            try {
                                rawCount = yoloEngine.DetectFrame(
                                    hwBuffer, detectionBuffer, confThreshold, nmsThreshold,
                                    rotateType, frameId, imageTimestampNs
                                )
                            } catch (e: Exception) {
                                Log.e(TAG, "Błąd DetectFrame frame=$frameId", e)
                            } finally {
                                hwBuffer.close()
                            }
                        }

                        updateProcessingFps()
                        val count = rawCount.coerceAtLeast(0)
                        if (rawCount < 0) Log.e(TAG, "Detekcja nie powiodła się: code=$rawCount frame=$frameId")

                        val topScore = if (rawCount >= 0) yoloEngine.GetTopScore() else 0f
                        val topClass = if (rawCount >= 0) yoloEngine.GetTopClass() else -1

                        if (frameId % 30L == 0L) {
                            Log.i(TAG, "Frame=$frameId imageTs=$imageTimestampNs FPS=" +
                                String.format(java.util.Locale.US, "%.1f", currentFps) +
                                " objects=$count top=" +
                                String.format(java.util.Locale.US, "%.3f", topScore) +
                                " class=$topClass")
                        }

                        val jsonPayload = buildJson(
                            count, detectionBuffer, currentFps, topScore, topClass, frameId, imageTimestampNs
                        )
                        try {
                            telemetryServer.sendDetections(jsonPayload)
                        } catch (e: Exception) {
                            Log.e(TAG, "Błąd wysyłania telemetrii", e)
                        }

                        // Vision test only: no actuator command is issued.
                        sendToSTM32(count, detectionBuffer, false)
                    } finally {
                        image.close()
                        isProcessingFrame.set(false)
                    }
                }
            }, cameraHandler)


            if (ActivityCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
                return
            }

            cameraManager.openCamera(selectedCameraId, object : CameraDevice.StateCallback() {
                override fun onOpened(camera: CameraDevice) {
                    cameraDevice = camera
                    createCameraCaptureSession()
                }

                override fun onDisconnected(camera: CameraDevice) {
                    camera.close()
                    cameraDevice = null
                    isCameraRunning.set(false)
                }

                override fun onError(camera: CameraDevice, error: Int) {
                    camera.close()
                    cameraDevice = null
                    isCameraRunning.set(false)
                    Log.e(TAG, "CameraDevice error code: $error")
                    telemetryServer.broadcastStatus("ERROR_CAMERA", "Błąd otwarcia Camera2: $error")
                }
            }, cameraHandler)

        } catch (e: Exception) {
            Log.e(TAG, "Błąd konfiguracji Camera2", e)
            telemetryServer.broadcastStatus("ERROR_CAMERA", "Błąd konfiguracji Camera2: ${e.message}")
        }
    }

    private fun createCameraCaptureSession() {
        val camera = cameraDevice ?: return
        val surface = imageReader?.surface ?: return

        try {
            val captureRequestBuilder = camera.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW).apply {
                addTarget(surface)
                set(CaptureRequest.CONTROL_AF_MODE, CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_PICTURE)
            }

            camera.createCaptureSession(listOf(surface), object : CameraCaptureSession.StateCallback() {
                override fun onConfigured(session: CameraCaptureSession) {
                    if (cameraDevice == null) return
                    captureSession = session
                    try {
                        val sensorTraceCallback=object:CameraCaptureSession.CaptureCallback(){
                            override fun onCaptureCompleted(session:CameraCaptureSession,request:CaptureRequest,result:TotalCaptureResult){
                                if(result.frameNumber%30L!=0L)return
                                val fpsRange=result.get(CaptureResult.CONTROL_AE_TARGET_FPS_RANGE)
                                Log.i(TAG,"CAMERA_SENSOR frame=${result.frameNumber} sensorTsNs=${result.get(CaptureResult.SENSOR_TIMESTAMP)} "+
                                    "exposureNs=${result.get(CaptureResult.SENSOR_EXPOSURE_TIME)} frameDurationNs=${result.get(CaptureResult.SENSOR_FRAME_DURATION)} "+
                                    "iso=${result.get(CaptureResult.SENSOR_SENSITIVITY)} aeFpsRange=${fpsRange?.lower}-${fpsRange?.upper} "+
                                    "aeState=${result.get(CaptureResult.CONTROL_AE_STATE)}")
                            }
                        }
                        session.setRepeatingRequest(captureRequestBuilder.build(),sensorTraceCallback,cameraHandler)
                        isCameraRunning.set(true)
                        Log.i(TAG, "Camera2 sesja skonfigurowana pomyślnie")
                        telemetryServer.broadcastStatus("STREAMING", "Camera2 → " + computeModeLabel(selectedComputeMode) + "; wyniki modelu są odczytywane do CPU")
                    } catch (e: CameraAccessException) {
                        Log.e(TAG, "Błąd setRepeatingRequest", e)
                    }
                }

                override fun onConfigureFailed(session: CameraCaptureSession) {
                    Log.e(TAG, "Konfiguracja sesji Camera2 nie powiodła się")
                    telemetryServer.broadcastStatus("ERROR_CAMERA", "Niepowodzenie konfiguracji sesji Camera2")
                }
            }, cameraHandler)
        } catch (e: Exception) {
            Log.e(TAG, "Błąd tworzenia CaptureSession", e)
        }
    }

    private fun startCameraThread() {
        if (cameraThread == null) {
            cameraThread = HandlerThread("Camera2BackgroundThread").apply {
                start()
                cameraHandler = Handler(looper)
            }
        }
    }

    private fun stopCameraThread() {
        cameraThread?.quitSafely()
        try {
            cameraThread?.join()
            cameraThread = null
            cameraHandler = null
        } catch (e: InterruptedException) {
            Log.e(TAG, "Błąd zatrzymywania wątku kamery", e)
        }
    }

    private fun updateProcessingFps() {
        val now = SystemClock.elapsedRealtime()
        if (fpsWindowStartMs == 0L) {
            fpsWindowStartMs = now
            fpsWindowFrames = 1
            return
        }
        fpsWindowFrames++
        val elapsedMs = now - fpsWindowStartMs
        if (elapsedMs >= 1000L) {
            currentFps = fpsWindowFrames * 1000f / elapsedMs
            fpsWindowFrames = 0
            fpsWindowStartMs = now
        }
    }

    private fun buildJson(
        count: Int, data: FloatArray, fpsVal: Float, topScore: Float, topClass: Int,
        frameId: Long, imageTimestampNs: Long
    ): String {
        val q = '"'
        val sb = StringBuilder(640)
        sb.append("{")
            .append(q).append("type").append(q).append(":").append(q).append("telemetry").append(q).append(",")
            .append(q).append("frameId").append(q).append(":").append(frameId).append(",")
            .append(q).append("imageTimestampNs").append(q).append(":").append(imageTimestampNs).append(",")
            .append(q).append("fps").append(q).append(":").append(String.format(java.util.Locale.US, "%.1f", fpsVal)).append(",")
            .append(q).append("topScore").append(q).append(":").append(String.format(java.util.Locale.US, "%.3f", topScore)).append(",")
            .append(q).append("topClass").append(q).append(":").append(topClass).append(",")
            .append(q).append("conf").append(q).append(":").append(String.format(java.util.Locale.US, "%.3f", confThreshold)).append(",")
            .append(q).append("count").append(q).append(":").append(count).append(",")
            .append(q).append("objects").append(q).append(":[")

        for (i in 0 until count) {
            val base = 1 + (i * 7)
            sb.append("{")
                .append(q).append("zone").append(q).append(":").append(data[base].toInt()).append(",")
                .append(q).append("class").append(q).append(":").append(data[base + 1].toInt()).append(",")
                .append(q).append("score").append(q).append(":").append(String.format(java.util.Locale.US, "%.3f", data[base + 2])).append(",")
                .append(q).append("x").append(q).append(":").append(String.format(java.util.Locale.US, "%.5f", data[base + 3])).append(",")
                .append(q).append("y").append(q).append(":").append(String.format(java.util.Locale.US, "%.5f", data[base + 4])).append(",")
                .append(q).append("w").append(q).append(":").append(String.format(java.util.Locale.US, "%.5f", data[base + 5])).append(",")
                .append(q).append("h").append(q).append(":").append(String.format(java.util.Locale.US, "%.5f", data[base + 6]))
            sb.append("}")
            if (i < count - 1) sb.append(",")
        }
        sb.append("]}")
        return sb.toString()
    }

    private fun sendToSTM32(count: Int, data: FloatArray, eStop: Boolean) {
        var critCount = 0
        var reactCount = 0
        var horizCount = 0

        for (i in 0 until count) {
            val zone = data[1 + i * 7].toInt()
            when (zone) {
                0 -> critCount++
                1 -> reactCount++
                2 -> horizCount++
            }
        }

        val packet = STM32UartProtocol.buildObstaclePacket(
            emergencyStop = eStop,
            criticalCount = critCount,
            reactionCount = reactCount,
            horizonCount = horizCount,
            nearestDistanceMm = if (critCount > 0) 150 else if (reactCount > 0) 650 else 2500,
            steerAngleDeg = 0
        )
        if (packet.isNotEmpty() && eStop) {
            Log.d(TAG, "STM32 E-STOP pakiet wysłany")
        }
    }

    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(baseContext, it) == PackageManager.PERMISSION_GRANTED
    }

    override fun onDestroy() {
        super.onDestroy()
        modeGeneration++
        isYoloReady.set(false)
        isCameraRunning.set(false)

        val readerToClose = imageReader
        try {
            readerToClose?.setOnImageAvailableListener(null, null)
            captureSession?.close()
            captureSession = null
            cameraDevice?.close()
            cameraDevice = null
            imageReader = null
        } catch (e: Exception) {
            Log.e(TAG, "Błąd zamykania zasobów Camera2", e)
        }

        stopCameraThread()

        if (::telemetryServer.isInitialized) {
            try {
                telemetryServer.stop()
            } catch (e: Exception) {
                Log.e(TAG, "Błąd zamykania TelemetryServer", e)
            }
        }

        // Drain in producer -> consumer order. A GPU task may enqueue a CPU
        // suffix until the GPU executor terminates, so CPU shutdown comes second.
        gpuFrameExecutor.shutdown()
        Thread {
            val gpuDrained = try {
                gpuFrameExecutor.awaitTermination(30, TimeUnit.SECONDS)
            } catch (e: InterruptedException) {
                Thread.currentThread().interrupt()
                false
            }
            if (!gpuDrained) {
                Log.e(TAG, "GPU frame executor did not drain; keeping native engine alive until process teardown")
                return@Thread
            }

            cpuFrameExecutor.shutdown()
            val cpuDrained = try {
                cpuFrameExecutor.awaitTermination(30, TimeUnit.SECONDS)
            } catch (e: InterruptedException) {
                Thread.currentThread().interrupt()
                false
            }
            if (!cpuDrained) {
                Log.e(TAG, "CPU frame executor did not drain; keeping native engine alive until process teardown")
                return@Thread
            }

            backgroundInitExecutor.shutdown()
            val initDrained = try {
                backgroundInitExecutor.awaitTermination(30, TimeUnit.SECONDS)
            } catch (e: InterruptedException) {
                Thread.currentThread().interrupt()
                false
            }
            if (!initDrained) {
                Log.e(TAG, "Initialization executor did not drain; keeping native engine alive until process teardown")
                return@Thread
            }

            // All three workers have released their acquired Camera2 Images.
            try {
                readerToClose?.close()
            } catch (e: Exception) {
                Log.e(TAG, "Błąd zamykania ImageReader po drenażu", e)
            }

            yoloEngine.ClearCache()
            yoloEngine.Destroy()
        }.start()
    }

    companion object {
        private const val TAG = "Mower_Vision_Core"
        private const val TARGET_SIZE_MIN = 320
        private const val TARGET_SIZE_MAX = 480
        private const val TARGET_SIZE_STEP = 32
        private const val REQUEST_CODE_PERMISSIONS = 10
        private val REQUIRED_PERMISSIONS = arrayOf(
            Manifest.permission.CAMERA,
            Manifest.permission.INTERNET
        )
    }
}
