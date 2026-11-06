package com.tencent.yolo11ncnn

import org.java_websocket.server.WebSocketServer
import org.java_websocket.handshake.ClientHandshake
import org.java_websocket.WebSocket
import java.net.InetSocketAddress
import android.util.Log

class TelemetryServer(
    port: Int = 8080,
    private val onCommandReceived: ((cmd: String, rawPayload: String) -> Unit)? = null
) : WebSocketServer(InetSocketAddress("0.0.0.0", port)) {

    @Volatile
    private var lastStatusJson: String? = null

    init {
        isReuseAddr = true
        connectionLostTimeout = 10
    }

    override fun onOpen(conn: WebSocket, handshake: ClientHandshake) {
        val q = '"'
        Log.i("TelemetryServer", "Dashboard connected: " + conn.remoteSocketAddress)
        val welcome = "{" + q + "type" + q + ":" + q + "status" + q + "," + q + "status" + q + ":" + q + "CONNECTED" + q + "," + q + "device" + q + ":" + q + "Samsung Galaxy S9+ (Mali-G72 MP18)" + q + "}"
        conn.send(welcome)
        lastStatusJson?.let {
            try { conn.send(it) } catch (_: Exception) {}
        }
    }

    override fun onClose(conn: WebSocket, code: Int, reason: String, remote: Boolean) {
        Log.i("TelemetryServer", "Dashboard disconnected: code=" + code)
    }

    override fun onMessage(conn: WebSocket, message: String) {
        try {
            val cmd = when {
                message.contains("EMERGENCY_STOP") -> "EMERGENCY_STOP"
                message.contains("RESET_ESTOP") -> "RESET_ESTOP"
                message.contains("SET_PARAMS") -> "SET_PARAMS"
                else -> "UNKNOWN"
            }
            onCommandReceived?.invoke(cmd, message)
        } catch (e: Exception) {
            Log.e("TelemetryServer", "Błąd parsowania", e)
        }
    }

    override fun onError(conn: WebSocket?, ex: Exception) {
        Log.e("TelemetryServer", "WebSocket błąd: " + ex.message)
    }

    override fun onStart() {
        Log.i("TelemetryServer", "WebSocket aktywny na porcie " + port + " (0.0.0.0:8080)")
    }

    fun broadcastStatus(status: String, detail: String) {
        val q = '"'
        val json = "{" + q + "type" + q + ":" + q + "status" + q + "," + q + "status" + q + ":" + q + status + q + "," + q + "detail" + q + ":" + q + detail + q + "," + q + "time" + q + ":" + System.currentTimeMillis() + "}"
        lastStatusJson = json
        try { broadcast(json) } catch (_: Exception) {}
    }

    fun sendDetections(jsonPayload: String) {
        try { broadcast(jsonPayload) } catch (_: Exception) {}
    }
}
