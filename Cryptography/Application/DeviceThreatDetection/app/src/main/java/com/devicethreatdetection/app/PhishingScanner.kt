
package com.devicethreatdetection.app

import android.content.Context
import org.json.JSONObject

data class PhishingScanResult(
    val prediction: String,
    val features: Map<String, Double>
)

class PhishingScanner(context: Context) {

    private val childrenLeft: IntArray
    private val childrenRight: IntArray
    private val features: IntArray
    private val thresholds: DoubleArray
    private val predictions: IntArray

    private val featureExtractor = PhishingFeatureExtractor(context)

    private val featureNames = listOf(
        "hostname_length",
        "dot_count",
        "hyphen_count",
        "at_count",
        "digit_count",
        "subdomain_count",
        "path_length",
        "brand_similarity",
        "exact_brand_match",
        "brand_substring_match"
    )

    init {
        val json = context.assets.open("phishing_tree.json")
            .bufferedReader()
            .use { it.readText() }

        val obj = JSONObject(json)

        childrenLeft = obj.getJSONArray("children_left").toIntArray()
        childrenRight = obj.getJSONArray("children_right").toIntArray()
        features = obj.getJSONArray("feature").toIntArray()
        thresholds = obj.getJSONArray("threshold").toDoubleArray()
        predictions = obj.getJSONArray("prediction").toIntArray()
    }

    // Original scanner interface is preserved.
    fun scanUrl(url: String): String {
        return scanUrlWithDetails(url).prediction
    }

    // Returns the original prediction and extracted feature values.
    fun scanUrlWithDetails(url: String): PhishingScanResult {

        require(url.isNotBlank()) {
            "Please enter a URL."
        }

        val input = featureExtractor.extractFeatures(url)

        require(input.size == featureNames.size) {
            "Unexpected number of extracted URL features."
        }

        var node = 0

        while (childrenLeft[node] != -1) {

            val featureIndex = features[node]

            require(featureIndex in input.indices) {
                "Invalid feature index in phishing tree."
            }

            node = if (input[featureIndex] <= thresholds[node]) {
                childrenLeft[node]
            } else {
                childrenRight[node]
            }
        }

        val prediction = if (predictions[node] == 1) {
            "Phishing"
        } else {
            "Legitimate"
        }

        val featureMap = featureNames.mapIndexed { index, name ->
            name to input[index]
        }.toMap()

        return PhishingScanResult(
            prediction = prediction,
            features = featureMap
        )
    }

    private fun org.json.JSONArray.toIntArray(): IntArray {
        return IntArray(length()) {
            getInt(it)
        }
    }

    private fun org.json.JSONArray.toDoubleArray(): DoubleArray {
        return DoubleArray(length()) {
            getDouble(it)
        }
    }
}
