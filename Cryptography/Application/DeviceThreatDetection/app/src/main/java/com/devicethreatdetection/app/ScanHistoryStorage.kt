
package com.devicethreatdetection.app

import android.content.Context
import org.json.JSONArray
import org.json.JSONObject

class ScanHistoryStorage(context: Context) {

    companion object {
        private const val PREFS_NAME = "device_threat_history"
        private const val HISTORY_KEY = "scan_history"
        private const val MAX_HISTORY = 50
    }

    private val preferences = context.getSharedPreferences(
        PREFS_NAME,
        Context.MODE_PRIVATE
    )

    fun saveHistory(history: List<ScanHistoryItem>) {

        val jsonArray = JSONArray()

        history.take(MAX_HISTORY).forEach { item ->

            val jsonObject = JSONObject().apply {
                put("module", item.module)
                put("input", item.input)
                put("result", item.result)
                put("scannedCount", item.scannedCount)
                put("threatCount", item.threatCount)
                put("timestamp", item.timestamp)
            }

            jsonArray.put(jsonObject)
        }

        preferences.edit()
            .putString(HISTORY_KEY, jsonArray.toString())
            .apply()
    }

    fun loadHistory(): List<ScanHistoryItem> {

        val savedData = preferences.getString(HISTORY_KEY, null)
            ?: return emptyList()

        return try {

            val jsonArray = JSONArray(savedData)
            val history = mutableListOf<ScanHistoryItem>()

            for (i in 0 until jsonArray.length()) {

                val item = jsonArray.getJSONObject(i)

                history.add(
                    ScanHistoryItem(
                        module = item.getString("module"),
                        input = item.optString("input", ""),
                        result = item.getString("result"),
                        scannedCount = item.getInt("scannedCount"),
                        threatCount = item.getInt("threatCount"),
                        timestamp = item.getString("timestamp")
                    )
                )
            }

            history.take(MAX_HISTORY)

        } catch (e: Exception) {

            emptyList()
        }
    }

    fun clearHistory() {

        preferences.edit()
            .remove(HISTORY_KEY)
            .apply()
    }
}
