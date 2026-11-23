
package com.devicethreatdetection.app

import java.io.InputStream

data class NetworkCsvScanSummary(
    val totalRows: Int,
    val benignRows: Int,
    val attackRows: Int,
    val rowsWithMissingValues: Int
)

class NetworkCsvScanner(
    private val scanner: NetworkIntrusionScanner
) {

    fun scanCsv(inputStream: InputStream): NetworkCsvScanSummary {

        inputStream.bufferedReader().use { reader ->

            val headerLine = reader.readLine()
                ?: throw IllegalArgumentException("CSV file is empty.")

            val originalHeaders = parseCsvLine(headerLine)
                .mapIndexed { index, name ->
                    if (index == 0) {
                        name.removePrefix("\uFEFF").trim()
                    } else {
                        name.trim()
                    }
                }

            require(originalHeaders.isNotEmpty()) {
                "CSV header is missing."
            }

            // Automatically make duplicate column names unique.
            // Example:
            // Fwd Header Length
            // Fwd Header Length
            //
            // Becomes:
            // Fwd Header Length
            // Fwd Header Length.1

            val headers = makeUniqueHeaders(originalHeaders)

            val featureNames = scanner.getFeatureNames()

            // Match model features with CSV column positions.
            val columnIndexes = featureNames.map { feature ->

                val index = headers.indexOf(feature)

                require(index >= 0) {
                    "Required feature missing from CSV: $feature"
                }

                index
            }

            var totalRows = 0
            var benignRows = 0
            var attackRows = 0
            var rowsWithMissingValues = 0

            var lineNumber = 1

            while (true) {

                val line = reader.readLine() ?: break
                lineNumber++

                if (line.isBlank()) continue

                val fields = parseCsvLine(line)

                require(fields.size == headers.size) {
                    "Invalid CSV row at line $lineNumber. " +
                            "Expected ${headers.size} columns but found ${fields.size}."
                }

                val featureValues = HashMap<String, Double?>()

                for (i in featureNames.indices) {

                    val featureName = featureNames[i]
                    val columnIndex = columnIndexes[i]

                    val rawValue = fields[columnIndex].trim()

                    val value = rawValue.toDoubleOrNull()
                        ?.takeIf { it.isFinite() }

                    featureValues[featureName] = value
                }

                val result = scanner.scanRecord(featureValues)

                totalRows++

                if (result.missingFeatureCount > 0) {
                    rowsWithMissingValues++
                }

                when (result.prediction) {

                    "Benign" -> benignRows++

                    "Attack" -> attackRows++

                    else -> throw IllegalStateException(
                        "Unexpected network prediction."
                    )
                }
            }

            require(totalRows > 0) {
                "CSV contains no network flow records."
            }

            return NetworkCsvScanSummary(
                totalRows = totalRows,
                benignRows = benignRows,
                attackRows = attackRows,
                rowsWithMissingValues = rowsWithMissingValues
            )
        }
    }

    // Generate unique column names while preserving their order.
    private fun makeUniqueHeaders(
        originalHeaders: List<String>
    ): List<String> {

        val usedNames = mutableSetOf<String>()
        val nextSuffix = mutableMapOf<String, Int>()

        return originalHeaders.map { name ->

            if (usedNames.add(name)) {

                name

            } else {

                var suffix = nextSuffix[name] ?: 1
                var candidate = "$name.$suffix"

                while (candidate in usedNames) {
                    suffix++
                    candidate = "$name.$suffix"
                }

                usedNames.add(candidate)
                nextSuffix[name] = suffix + 1

                candidate
            }
        }
    }

    private fun parseCsvLine(line: String): List<String> {

        val fields = ArrayList<String>()
        val current = StringBuilder()

        var insideQuotes = false
        var i = 0

        while (i < line.length) {

            val character = line[i]

            when {

                character == '"' && insideQuotes &&
                        i + 1 < line.length &&
                        line[i + 1] == '"' -> {

                    current.append('"')
                    i++
                }

                character == '"' -> {
                    insideQuotes = !insideQuotes
                }

                character == ',' && !insideQuotes -> {
                    fields.add(current.toString())
                    current.clear()
                }

                else -> {
                    current.append(character)
                }
            }

            i++
        }

        require(!insideQuotes) {
            "Invalid CSV: unmatched quotation mark."
        }

        fields.add(current.toString())

        return fields
    }
}
