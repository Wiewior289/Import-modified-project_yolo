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
import android.widget.TextView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import java.util.concurrent.ArrayBlockingQueue
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.RejectedExecutionException
import java.util.concurrent.atomic.AtomicBoolean

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
    private lateinit var telemetryServer: TelemetryServer
    private lateinit var backgroundInitExecutor: ExecutorService
    private lateinit var gpuPrefixExecutor: ExecutorService
    private lateinit var cpuSuffixExecutor: ExecutorService

    private class FrameSlot {
        val detectionBuffer = FloatArray(1 + 64 * 7)
    }

    // Two bounded slots: one may be in CPU suffix while the other is in GPU prefix.
    private val frameSlots = ArrayBlockingQueue<FrameSlot>(2).apply {
        add(FrameSlot())
        add(FrameSlot())
    }
    private val frameAdmissionLock = Any()

    // Camera2 background thread & components
    private var cameraThread: HandlerThread? = null
    private var cameraHandler: Handler? = null
    private var cameraDevice: CameraDevice? = null
    private var captureSession: CameraCaptureSession? = null
    private var imageReader: ImageReader? = null
    private var sensorOrientation: Int = 90

    @Volatile private var confThreshold: Float = 0.25f
    @Volatile private var nmsThreshold: Float = 0.45f

    private val isYoloReady = AtomicBoolean(false)
    private val isCameraRunning = AtomicBoolean(false)

    private var frameSequence: Long = 0L
    private var fpsWindowStartMs: Long = 0L
    private var fpsWindowFrames: Int = 0
    private var currentFps: Float = 0f

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        backgroundInitExecutor = Executors.newSingleThreadExecutor()
        gpuPrefixExecutor = Executors.newSingleThreadExecutor()
        cpuSuffixExecutor = Executors.newSingleThreadExecutor()

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
        val labels = listOf("CPU", "GPU Vulkan", "GPU Vulkan + AHB")
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(20, 16, 20, 8)
        }
        val label = TextView(this).apply {
            text = "Tryb obliczeniowy NCNN"
            textSize = 16f
        }
        val spinner = Spinner(this)
        val adapter = ArrayAdapter(this, android.R.layout.simple_spinner_item, labels)
        adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
        spinner.adapter = adapter
        spinner.setSelection(selectedComputeMode)
        spinner.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
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
        root.addView(label, LinearLayout.LayoutParams(
            LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT
        ))
        root.addView(spinner, LinearLayout.LayoutParams(
            LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT
        ))
        setContentView(root)
    }

    private fun computeModeLabel(mode: Int): String = when (mode) {
        Yolo11Ncnn.COMPUTE_CPU -> "CPU"
        Yolo11Ncnn.COMPUTE_VULKAN -> "GPU Vulkan"
        Yolo11Ncnn.COMPUTE_VULKAN_AHB -> "GPU Vulkan + AHB"
        else -> "UNKNOWN($mode)"
    }

    private fun closeCameraForModeSwitch() {
        try {
            captureSession?.close()
            captureSession = null
            cameraDevice?.close()
            cameraDevice = null
        } catch (e: Exception) {
            Log.e(TAG, "Błąd zatrzymywania kamery przy zmianie backendu", e)
        }
    }

    private fun switchComputeMode(newMode: Int) {
        val generation: Long
        synchronized(frameAdmissionLock) {
            if (newMode !in Yolo11Ncnn.COMPUTE_CPU..Yolo11Ncnn.COMPUTE_VULKAN_AHB ||
                newMode == selectedComputeMode) return

            selectedComputeMode = newMode
            generation = ++modeGeneration
            isYoloReady.set(false)
            isCameraRunning.set(false)
            closeCameraForModeSwitch()
        }
        telemetryServer.broadcastStatus(
            "COMPUTE_MODE_SWITCH",
            "Przełączanie backendu na " + computeModeLabel(newMode)
        )

        // GPU barrier runs after every admitted prefix. It then enqueues a CPU
        // barrier after all suffix jobs those prefixes produced. Only then may
        // the model and AHB cache be destroyed/reloaded.
        gpuPrefixExecutor.execute {
            cpuSuffixExecutor.execute {
                backgroundInitExecutor.execute {
                    fpsWindowStartMs = 0L
            fpsWindowFrames = 0
            currentFps = 0f
            yoloEngine.Destroy()
            val startedAt = SystemClock.elapsedRealtime()
            val initialized = try {
                yoloEngine.Init(assets, Yolo11Ncnn.MODEL_YOLO11N, 320, newMode)
            } catch (e: Exception) {
                Log.e(TAG, "Błąd inicjalizacji backendu " + computeModeLabel(newMode), e)
                false
            }
            val durationMs = SystemClock.elapsedRealtime() - startedAt

            runOnUiThread {
                if (generation != modeGeneration) return@runOnUiThread
                // The executor has drained the prior frame. Detach the old reader
                // callback before closing the reader and opening the next session.
                imageReader?.setOnImageAvailableListener(null, null)
                imageReader?.close()
                imageReader = null
                if (initialized) {
                    isYoloReady.set(true)
                    Log.i(TAG, "Backend " + computeModeLabel(newMode) + " gotowy w " + durationMs + "ms")
                    telemetryServer.broadcastStatus(
                        "COMPUTE_MODE_READY",
                        computeModeLabel(newMode) + " gotowy (" + durationMs + "ms)"
                    )
                    startCamera2()
                } else {
                    isYoloReady.set(false)
                    telemetryServer.broadcastStatus(
                        "ERROR_COMPUTE_MODE",
                        "Nie udało się uruchomić backendu " + computeModeLabel(newMode)
                    )
                }
                }
            }
        }
    }

    private fun onImageAvailable(reader: ImageReader) {
        synchronized(frameAdmissionLock) {
            if (!isYoloReady.get()) {
                reader.acquireLatestImage()?.close()
                return@synchronized
            }

            val image = reader.acquireLatestImage() ?: return@synchronized
            val slot = frameSlots.poll()
            if (slot == null) {
                image.close()
                return@synchronized
            }

            val frameId = ++frameSequence
            val imageTimestampNs = image.timestamp
            val rotateType = when (sensorOrientation) {
                90 -> 6
                180 -> 3
                270 -> 8
                else -> 1
            }
            if (frameId % 30L == 0L) {
                Log.i(TAG, "CAMERA_IMAGE frame=$frameId imageTs=$imageTimestampNs " +
                    "size=${image.width}x${image.height} crop=${image.cropRect}")
            }

            if (selectedComputeMode == Yolo11Ncnn.COMPUTE_VULKAN_AHB) {
                launchGpuPrefix(slot, image, rotateType, frameId, imageTimestampNs)
            } else {
                launchSynchronousFrame(slot, image, rotateType, frameId, imageTimestampNs)
            }
        }
    }

    private fun launchGpuPrefix(
        slot: FrameSlot,
        image: android.media.Image,
        rotateType: Int,
        frameId: Long,
        imageTimestampNs: Long
    ) {
        try {
            gpuPrefixExecutor.execute {
                var imageClosed = false
                var token = -5L
                try {
                    val hardwareBuffer = image.hardwareBuffer
                    if (hardwareBuffer == null) {
                        token = -4L
                    } else {
                        try {
                            // Native returns only after the GPU-prefix fence. The
                            // camera Image therefore stays alive throughout import,
                            // preprocessing, prefix execution and boundary readback.
                            token = yoloEngine.BeginFrame(
                                hardwareBuffer, rotateType, frameId, imageTimestampNs
                            )
                        } finally {
                            hardwareBuffer.close()
                        }
                    }

                    image.close()
                    imageClosed = true
                    enqueueCpuCompletion(slot, token, frameId, imageTimestampNs)
                } catch (e: Exception) {
                    Log.e(TAG, "Błąd GPU prefix frame=$frameId", e)
                    if (token > 0L) {
                        try { yoloEngine.CancelFrame(token) } catch (_: Exception) {}
                    }
                    if (!imageClosed) {
                        try { image.close() } catch (_: Exception) {}
                    }
                    enqueueCpuCompletion(slot, -5L, frameId, imageTimestampNs)
                } finally {
                    if (!imageClosed) {
                        try { image.close() } catch (_: Exception) {}
                    }
                }
            }
        } catch (e: RejectedExecutionException) {
            Log.e(TAG, "GPU prefix executor rejected frame=$frameId", e)
            try { image.close() } catch (_: Exception) {}
            releaseFrameSlot(slot)
        }
    }

    private fun enqueueCpuCompletion(
        slot: FrameSlot,
        prefixToken: Long,
        frameId: Long,
        imageTimestampNs: Long
    ) {
        try {
            cpuSuffixExecutor.execute {
                slot.detectionBuffer.fill(0f)
                val rawCount = if (prefixToken > 0L) {
                    try {
                        yoloEngine.FinishFrame(
                            prefixToken, slot.detectionBuffer, confThreshold, nmsThreshold
                        )
                    } catch (e: Exception) {
                        Log.e(TAG, "Błąd CPU suffix frame=$frameId", e)
                        try { yoloEngine.CancelFrame(prefixToken) } catch (_: Exception) {}
                        -5
                    }
                } else {
                    prefixToken.toInt()
                }

                try {
                    publishFrameResult(slot, rawCount, frameId, imageTimestampNs)
                } catch (e: Exception) {
                    Log.e(TAG, "Błąd publikacji wyniku frame=$frameId", e)
                } finally {
                    releaseFrameSlot(slot)
                }
            }
        } catch (e: RejectedExecutionException) {
            Log.e(TAG, "CPU suffix executor rejected frame=$frameId", e)
            if (prefixToken > 0L) {
                try { yoloEngine.CancelFrame(prefixToken) } catch (_: Exception) {}
            }
            releaseFrameSlot(slot)
        }
    }

    private fun launchSynchronousFrame(
        slot: FrameSlot,
        image: android.media.Image,
        rotateType: Int,
        frameId: Long,
        imageTimestampNs: Long
    ) {
        try {
            cpuSuffixExecutor.execute {
                slot.detectionBuffer.fill(0f)
                var rawCount = -5
                try {
                    val hardwareBuffer = image.hardwareBuffer
                    if (hardwareBuffer == null) {
                        rawCount = -4
                    } else {
                        try {
                            rawCount = yoloEngine.DetectFrame(
                                hardwareBuffer, slot.detectionBuffer, confThreshold, nmsThreshold,
                                rotateType, frameId, imageTimestampNs
                            )
                        } finally {
                            hardwareBuffer.close()
                        }
                    }
                } catch (e: Exception) {
                    Log.e(TAG, "Błąd DetectFrame frame=$frameId", e)
                } finally {
                    try { image.close() } catch (_: Exception) {}
                }

                try {
                    publishFrameResult(slot, rawCount, frameId, imageTimestampNs)
                } catch (e: Exception) {
                    Log.e(TAG, "Błąd publikacji wyniku frame=$frameId", e)
                } finally {
                    releaseFrameSlot(slot)
                }
            }
        } catch (e: RejectedExecutionException) {
            Log.e(TAG, "CPU executor rejected frame=$frameId", e)
            try { image.close() } catch (_: Exception) {}
            releaseFrameSlot(slot)
        }
    }

    private fun publishFrameResult(
        slot: FrameSlot,
        rawCount: Int,
        frameId: Long,
        imageTimestampNs: Long
    ) {
        val buffer = slot.detectionBuffer
        updateProcessingFps()
        val count = rawCount.coerceAtLeast(0)
        if (rawCount < 0) {
            Log.e(TAG, "Detekcja nie powiodła się: code=$rawCount frame=$frameId")
        }

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
            count, buffer, currentFps, topScore, topClass, frameId, imageTimestampNs
        )
        try {
            telemetryServer.sendDetections(jsonPayload)
        } catch (e: Exception) {
            Log.e(TAG, "Błąd wysyłania telemetrii", e)
        }

        // Vision test only: no actuator command is issued.
        sendToSTM32(count, buffer, false)
    }

    private fun releaseFrameSlot(slot: FrameSlot) {
        if (!frameSlots.offer(slot)) {
            Log.e(TAG, "Frame slot double-release detected")
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
        val generationAtStart = modeGeneration
        backgroundInitExecutor.execute {
            Log.i(TAG, "Ładowanie YOLO11 backend=" + computeModeLabel(modeAtStart) + " w tle...")
            telemetryServer.broadcastStatus(
                "INIT_YOLO",
                "Inicjalizacja YOLO11: " + computeModeLabel(modeAtStart)
            )

            val startTime = SystemClock.elapsedRealtime()
            val initSuccess = try {
                yoloEngine.Init(assets, Yolo11Ncnn.MODEL_YOLO11N, 320, modeAtStart)
            } catch (e: Exception) {
                Log.e(TAG, "Błąd podczas init backendu " + computeModeLabel(modeAtStart), e)
                false
            }
            val loadDuration = SystemClock.elapsedRealtime() - startTime

            runOnUiThread {
                if (generationAtStart != modeGeneration) return@runOnUiThread
                if (initSuccess) {
                    Log.i(TAG, "YOLO11 " + computeModeLabel(modeAtStart) + " Init sukces w " + loadDuration + "ms")
                    isYoloReady.set(true)
                    telemetryServer.broadcastStatus(
                        "YOLO_READY",
                        computeModeLabel(modeAtStart) + " gotowy (" + loadDuration + "ms)"
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
                onImageAvailable(reader)
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
        synchronized(frameAdmissionLock) {
            modeGeneration++
            isYoloReady.set(false)
            isCameraRunning.set(false)
            imageReader?.setOnImageAvailableListener(null, null)
        }

        try {
            captureSession?.close()
            captureSession = null
            cameraDevice?.close()
            cameraDevice = null
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

        // Drain order matters: GPU tasks may still enqueue CPU suffix tasks.
        // The final CPU barrier is therefore placed only after the GPU barrier.
        try {
            gpuPrefixExecutor.execute {
                cpuSuffixExecutor.execute {
                    backgroundInitExecutor.execute {
                        try {
                            yoloEngine.ClearCache()
                            yoloEngine.Destroy()
                        } catch (e: Exception) {
                            Log.e(TAG, "Błąd kończenia silnika YOLO", e)
                        } finally {
                            try { imageReader?.close() } catch (_: Exception) {}
                            imageReader = null
                            gpuPrefixExecutor.shutdown()
                            cpuSuffixExecutor.shutdown()
                            backgroundInitExecutor.shutdown()
                        }
                    }
                }
            }
        } catch (e: RejectedExecutionException) {
            Log.e(TAG, "Nie udało się zlecić bezpiecznego zamknięcia potoku", e)
        }
    }

    companion object {
        private const val TAG = "Mower_Vision_Core"
        private const val REQUEST_CODE_PERMISSIONS = 10
        private val REQUIRED_PERMISSIONS = arrayOf(
            Manifest.permission.CAMERA,
            Manifest.permission.INTERNET
        )
    }
}
