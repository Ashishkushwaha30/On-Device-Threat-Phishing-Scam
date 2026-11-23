
package com.devicethreatdetection.app

import android.content.Context
import org.json.JSONArray
import org.json.JSONObject
import java.util.Locale
import java.util.regex.Pattern
import kotlin.math.ln
import kotlin.math.sqrt

data class SmsScanResult(
    val prediction: String,
    val spamProbability: Double,
    val hamProbability: Double
)

class SmsSpamScanner(context: Context) {

    private val vocabulary = HashMap<String, Int>()
    private val idf = DoubleArray(10000)

    private val childrenLeft: IntArray
    private val childrenRight: IntArray
    private val features: IntArray
    private val thresholds: DoubleArray
    private val predictions: IntArray

    private val tokenizer = Pattern.compile("\\b\\w+\\b")

    init {

        val json = context.assets
            .open("sms_model_android.json")
            .bufferedReader()
            .use { it.readText() }

        val obj = JSONObject(json)

        // Load vocabulary
        val vocabularyObject = obj.getJSONObject("vocabulary")
        val keys = vocabularyObject.keys()

        while (keys.hasNext()) {
            val word = keys.next()
            vocabulary[word] = vocabularyObject.getInt(word)
        }

        require(vocabulary.size == 10000) {
            "Unexpected SMS vocabulary size."
        }

        // Load IDF values
        val idfArray = obj.getJSONArray("idf")

        require(idfArray.length() == idf.size) {
            "Unexpected IDF vector size."
        }

        for (i in idf.indices) {
            idf[i] = idfArray.getDouble(i)
        }

        // Verify TF-IDF configuration
        val config = obj.getJSONObject("tfidf_config")

        require(config.getString("analyzer") == "word")
        require(config.getBoolean("lowercase"))
        require(config.getBoolean("sublinear_tf"))
        require(config.getString("norm") == "l2")

        // Load Decision Tree
        childrenLeft = obj.getJSONArray("children_left").toIntArray()
        childrenRight = obj.getJSONArray("children_right").toIntArray()
        features = obj.getJSONArray("feature").toIntArray()
        thresholds = obj.getJSONArray("threshold").toDoubleArray()
        predictions = obj.getJSONArray("prediction").toIntArray()

        require(
            childrenLeft.size == childrenRight.size &&
                    childrenLeft.size == features.size &&
                    childrenLeft.size == thresholds.size &&
                    childrenLeft.size == predictions.size
        ) {
            "Invalid Decision Tree data."
        }
    }

    fun scanSms(message: String): SmsScanResult {

        require(message.isNotBlank()) {
            "Please enter an SMS message."
        }

        // Step 1: Convert text to lowercase
        val text = message.lowercase(Locale.ROOT)

        // Step 2: Tokenize using the trained token pattern
        val matcher = tokenizer.matcher(text)

        val tokens = ArrayList<String>()

        while (matcher.find()) {
            tokens.add(matcher.group())
        }

        // Step 3: Generate unigram and bigram term frequencies
        val termFrequency = HashMap<Int, Int>()

        fun addTerm(term: String) {
            val index = vocabulary[term] ?: return

            termFrequency[index] =
                (termFrequency[index] ?: 0) + 1
        }

        for (i in tokens.indices) {

            // Unigram
            addTerm(tokens[i])

            // Bigram
            if (i + 1 < tokens.size) {
                addTerm(tokens[i] + " " + tokens[i + 1])
            }
        }

        // Step 4: Apply sublinear TF and IDF
        val tfidf = HashMap<Int, Double>()

        for ((index, count) in termFrequency) {

            val sublinearTf = 1.0 + ln(count.toDouble())

            tfidf[index] = sublinearTf * idf[index]
        }

        // Step 5: L2 normalization
        var squaredSum = 0.0

        for (value in tfidf.values) {
            squaredSum += value * value
        }

        val norm = sqrt(squaredSum)

        if (norm > 0.0) {
            for ((index, value) in tfidf.toMap()) {
                tfidf[index] = value / norm
            }
        }

        // Step 6: Traverse the Decision Tree
        var node = 0

        while (childrenLeft[node] != -1) {

            val featureIndex = features[node]

            require(featureIndex in 0 until idf.size) {
                "Invalid feature index in SMS Decision Tree."
            }

            val featureValue = tfidf[featureIndex] ?: 0.0

            node = if (featureValue <= thresholds[node]) {
                childrenLeft[node]
            } else {
                childrenRight[node]
            }

            require(node in childrenLeft.indices) {
                "Invalid Decision Tree node."
            }
        }

        // Step 7: Return predicted class
        val prediction = predictions[node]

        return when (prediction) {

            0 -> SmsScanResult(
                prediction = "Ham (Legitimate)",
                spamProbability = 0.0,
                hamProbability = 1.0
            )

            1 -> SmsScanResult(
                prediction = "Spam",
                spamProbability = 1.0,
                hamProbability = 0.0
            )

            else -> throw IllegalStateException(
                "Unexpected SMS prediction class: $prediction"
            )
        }
    }

    private fun JSONArray.toIntArray(): IntArray {
        return IntArray(length()) {
            getInt(it)
        }
    }

    private fun JSONArray.toDoubleArray(): DoubleArray {
        return DoubleArray(length()) {
            getDouble(it)
        }
    }
}
