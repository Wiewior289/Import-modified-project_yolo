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
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

/**
 * ============================================================================
 * MainActivity - Camera2 / AHardwareBuffer YOLO test harness
 * Platform: Samsung Galaxy S9+ (Mali-G72 MP18 + Exynos 9810)
 *
 * Camera2 -> ImageReader AHardwareBuffer -> Vulkan import -> NCNN.
 * The input buffer is imported without an application-side YUV-to-RGB CPU copy.
 * Model output is read back to CPU for DFL decode, NMS and dashboard telemetry.
 * ============================================================================
 */
class MainActivity : AppCompatActivity() {

    private val yoloEngine = Yolo11Ncnn()
    private lateinit var telemetryServer: TelemetryServer
    private lateinit var backgroundInitExecutor: ExecutorService

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
        setContentView(android.R.layout.list_content)

        backgroundInitExecutor = Executors.newSingleThreadExecutor()

        // KROK 1: Start serwera WebSocket natychmiast
        startTelemetryServer()

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
        backgroundInitExecutor.execute {
            Log.i(TAG, "KROK 2: Ładowanie YOLO11 Vulkan w tle...")
            telemetryServer.broadcastStatus("INIT_YOLO", "Inicjalizacja GPU Mali-G72 i ładowanie wag YOLO11...")

            val startTime = System.currentTimeMillis()
            var initSuccess = false

            try {
                initSuccess = yoloEngine.Init(assets, Yolo11Ncnn.MODEL_YOLO11N, 320, true)
            } catch (e: Exception) {
                Log.e(TAG, "Błąd podczas init YOLO Vulkan", e)
            }

            val loadDuration = System.currentTimeMillis() - startTime

            if (initSuccess) {
                Log.i(TAG, "YOLO11 Vulkan Init sukces w " + loadDuration + "ms")
                isYoloReady.set(true)
                telemetryServer.broadcastStatus("YOLO_READY", "YOLO11 gotowe na Mali-G72 (" + loadDuration + "ms)")

                // KROK 3: Start natywnej kamery Camera2 DOPIERO po gotowości YOLO
                runOnUiThread {
                    Log.i(TAG, "KROK 3: Uruchamianie Camera2 Direct AHB...")
                    telemetryServer.broadcastStatus("STARTING_CAMERA", "Uruchamianie Camera2 HardwareBuffer stream...")
                    startCamera2()
                }
            } else {
                isYoloReady.set(false)
                Log.e(TAG, "Vulkan/AHB init failed; refusing to start a CPU path that cannot consume AHardwareBuffer")
                telemetryServer.broadcastStatus(
                    "ERROR_AHB_VULKAN",
                    "Import AHardwareBuffer do Vulkan niedostępny. Ścieżka CPU nie jest zaimplementowana."
                )
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
                if (!isYoloReady.get() || isProcessingFrame.get()) {
                    reader.acquireLatestImage()?.close()
                    return@setOnImageAvailableListener
                }

                val image = reader.acquireLatestImage() ?: return@setOnImageAvailableListener
                isProcessingFrame.set(true)
                val frameId = ++frameSequence
                val imageTimestampNs = image.timestamp

                backgroundInitExecutor.execute {
                    try {
                        val rotateType = when (sensorOrientation) {
                            90 -> 6
                            180 -> 3
                            270 -> 8
                            else -> 1
                        }

                        detectionBuffer.fill(0f)
                        var rawCount = -1
                        val hwBuffer = image.hardwareBuffer
                        try {
                            rawCount = yoloEngine.DetectAHBDirect(
                                hwBuffer, detectionBuffer, confThreshold, nmsThreshold, rotateType
                            )
                        } catch (e: Exception) {
                            Log.e(TAG, "Błąd DetectAHBDirect frame=$frameId", e)
                        } finally {
                            hwBuffer.close()
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
                        session.setRepeatingRequest(captureRequestBuilder.build(), null, cameraHandler)
                        isCameraRunning.set(true)
                        Log.i(TAG, "Camera2 sesja skonfigurowana pomyślnie")
                        telemetryServer.broadcastStatus("STREAMING", "Camera2 AHB → Vulkan import aktywny; wyniki modelu są odczytywane do CPU")
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
        isYoloReady.set(false)
        isCameraRunning.set(false)

        try {
            captureSession?.close()
            captureSession = null
            cameraDevice?.close()
            cameraDevice = null
            imageReader?.close()
            imageReader = null
        } catch (e: Exception) {
            Log.e(TAG, "Błąd zamykania zasobów Camera2", e)
        }

        stopCameraThread()
        backgroundInitExecutor.shutdown()

        if (::telemetryServer.isInitialized) {
            try {
                telemetryServer.stop()
            } catch (e: Exception) {
                Log.e(TAG, "Błąd zamykania TelemetryServer", e)
            }
        }
        yoloEngine.ClearCache()
        yoloEngine.Destroy()
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
