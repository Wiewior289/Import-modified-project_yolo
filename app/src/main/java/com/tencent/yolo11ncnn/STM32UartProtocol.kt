package com.tencent.yolo11ncnn

import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * ============================================================================
 * High-Speed Binary Telemetry & Control Protocol for STM32F405 Actuator
 * Samsung Galaxy S9+ USB-UART / CAN Bridge (Baudrate: 115200 / 921600 bps)
 *
 * Transmits real-time obstacle vectors and hardware Emergency Stop triggers
 * with strict CRC16 error validation for industrial safety compliance.
 * ============================================================================
 */
object STM32UartProtocol {

    private const val HEADER_0: Byte = 0xAA.toByte()
    private const val HEADER_1: Byte = 0x55.toByte()
    private const val MSG_OBSTACLE_TELEMETRY: Byte = 0x10.toByte()

    // Pre-allocated transmission buffer (Zero-Allocation on communication thread)
    private val txBuffer = ByteBuffer.allocate(24).order(ByteOrder.LITTLE_ENDIAN)

    /**
     * Builds binary packet for STM32F405 cutter & drive controller:
     * [0xAA, 0x55, MSG_ID, LEN, E_STOP, CRIT_CNT, REACT_CNT, HORIZ_CNT, MIN_DIST_MM, STEER_ANGLE_DEG, CRC16_L, CRC16_H]
     */
    @Synchronized
    fun buildObstaclePacket(
        emergencyStop: Boolean,
        criticalCount: Int,
        reactionCount: Int,
        horizonCount: Int,
        nearestDistanceMm: Int,
        steerAngleDeg: Short
    ): ByteArray {
        txBuffer.clear()

        // 1. Packet Preamble
        txBuffer.put(HEADER_0)
        txBuffer.put(HEADER_1)
        txBuffer.put(MSG_OBSTACLE_TELEMETRY)
        txBuffer.put(8.toByte()) // Payload Length: 8 bytes

        // 2. Safety State & Zone Obstacle Distribution
        val eStopByte: Byte = if (emergencyStop) 0x01.toByte() else 0x00.toByte()
        txBuffer.put(eStopByte)
        txBuffer.put(criticalCount.coerceIn(0, 255).toByte())
        txBuffer.put(reactionCount.coerceIn(0, 255).toByte())
        txBuffer.put(horizonCount.coerceIn(0, 255).toByte())

        // 3. Vector to Nearest Obstacle for Trajectory Deflection
        txBuffer.putShort(nearestDistanceMm.coerceIn(0, 65535).toShort())
        txBuffer.putShort(steerAngleDeg)

        // 4. Compute CRC16 CCITT
        val payloadLength = 12 // 4 header + 8 payload
        val crc = computeCrc16(txBuffer.array(), 2, payloadLength)
        txBuffer.putShort(crc.toShort())

        val packet = ByteArray(14)
        System.arraycopy(txBuffer.array(), 0, packet, 0, 14)
        return packet
    }

    private fun computeCrc16(data: ByteArray, offset: Int, length: Int): Int {
        var crc = 0xFFFF
        for (i in offset until length) {
            crc = (crc ushr 8) xor CRC16_TABLE[(crc xor (data[i].toInt() and 0xFF)) and 0xFF]
        }
        return crc and 0xFFFF
    }

    private val CRC16_TABLE = IntArray(256).apply {
        for (i in 0 until 256) {
            var curr = i
            for (j in 0 until 8) {
                curr = if ((curr and 1) != 0) (curr ushr 1) xor 0x8408 else curr ushr 1
            }
            this[i] = curr
        }
    }
}
