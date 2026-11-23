
package com.devicethreatdetection.app

import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

@Composable
fun NetworkDetectionScreen(
    activity: MainActivity,
    onBack: () -> Unit = {}
) {

    val context = LocalContext.current
    val scope = rememberCoroutineScope()

    val scanner = remember {
        NetworkCsvScanner(
            NetworkIntrusionScanner(context.applicationContext)
        )
    }

    var result by remember {
        mutableStateOf<NetworkCsvScanSummary?>(null)
    }

    var errorMessage by remember {
        mutableStateOf<String?>(null)
    }

    var explanation by remember {
        mutableStateOf("")
    }

    var isScanning by remember {
        mutableStateOf(false)
    }

    var isExplaining by remember {
        mutableStateOf(false)
    }

    var selectedFile by remember {
        mutableStateOf<String?>(null)
    }

    val filePicker = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.OpenDocument()
    ) { uri ->

        if (uri != null) {

            selectedFile = uri.lastPathSegment
            result = null
            explanation = ""
            errorMessage = null
            isScanning = true

            scope.launch {

                try {

                    val scanResult = withContext(Dispatchers.IO) {

                        val inputStream =
                            context.contentResolver.openInputStream(uri)
                                ?: throw IllegalStateException(
                                    "Unable to open selected CSV file."
                                )

                        inputStream.use {
                            scanner.scanCsv(it)
                        }
                    }

                    result = scanResult

                    // Record the successful network scan
                    activity.recordScan(
                        module = "Network",
                        input = """
                            CSV File: ${selectedFile ?: "Unknown file"}
                            Total flows: ${scanResult.totalRows}
                            Benign flows: ${scanResult.benignRows}
                            Attack flows: ${scanResult.attackRows}
                            Rows with missing or invalid values: ${scanResult.rowsWithMissingValues}
                        """.trimIndent(),
                        result = "${scanResult.attackRows} attack flows out of ${scanResult.totalRows}",
                        scannedCount = scanResult.totalRows,
                        threatCount = scanResult.attackRows
                    )

                    // Generate Llama explanation if the model is loaded
                    if (activity.isLlamaLoaded()) {

                        isExplaining = true
                        explanation = "Generating AI explanation..."

                        val prompt = """
                            <|begin_of_text|><|start_header_id|>system<|end_header_id|>

                            You are a cybersecurity assistant explaining network intrusion detection results to ordinary smartphone users.

                            The network classifications have already been produced by a Decision Tree model. These classifications are final. You must not change them.

                            Explain the aggregate network scan results in simple English.

                            Explain:
                            - How many network flows were analysed.
                            - How many were classified as benign.
                            - How many were classified as attacks.
                            - Whether any rows contained missing or invalid feature values.

                            IMPORTANT RULES:
                            Do not invent attack types, IP addresses, ports, suspicious applications, attack sources, or reasons for individual predictions.
                            Do not claim that you inspected live network traffic.
                            Do not claim that the detected attacks are confirmed real-world attacks.
                            Explain that the attack count represents flows flagged by the trained model and should be investigated further.
                            If missing or invalid values were present, explain that the scanner used median imputation for those features.

                            Give practical advice to review flagged network flows and investigate unusual activity.

                            Write 4 to 6 short sentences in one paragraph.
                            Use simple English. No headings or bullet points.

                            <|eot_id|><|start_header_id|>user<|end_header_id|>

                            Network scan summary:

                            Total flows: ${scanResult.totalRows}
                            Benign flows: ${scanResult.benignRows}
                            Attack flows: ${scanResult.attackRows}
                            Rows with missing or invalid values: ${scanResult.rowsWithMissingValues}

                            Explain these exact results without changing any numbers or classifications.

                            <|eot_id|><|start_header_id|>assistant<|end_header_id|>
                        """.trimIndent()

                        Thread {

                            try {

                                val aiResult =
                                    activity.generateLlamaResponse(prompt)

                                activity.runOnUiThread {

                                    explanation = aiResult
                                    isExplaining = false
                                }

                            } catch (e: Exception) {

                                activity.runOnUiThread {

                                    explanation =
                                        "AI explanation failed: ${e.message}"

                                    isExplaining = false
                                }
                            }

                        }.start()

                    } else {

                        explanation =
                            "Load Llama from the AI Assistant module to generate a network scan explanation."
                    }

                } catch (e: Exception) {

                    errorMessage = e.message
                        ?: "An unexpected error occurred."

                } finally {

                    isScanning = false
                }
            }
        }
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(20.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp)
    ) {

        Button(onClick = onBack) {
            Text("Back")
        }

        Text(
            text = "Network Intrusion Detection",
            style = MaterialTheme.typography.headlineSmall
        )

        Text(
            "Select a CICFlowMeter CSV file to analyse network traffic flows and detect potential intrusions."
        )

        Button(
            onClick = {
                filePicker.launch(arrayOf("*/*"))
            },
            enabled = !isScanning && !isExplaining,
            modifier = Modifier.fillMaxWidth()
        ) {
            Text("Select Network CSV File")
        }

        selectedFile?.let {
            Text(
                text = "Selected file: $it",
                style = MaterialTheme.typography.bodySmall
            )
        }

        if (isScanning) {

            Card(
                modifier = Modifier.fillMaxWidth()
            ) {

                Row(
                    modifier = Modifier.padding(16.dp),
                    horizontalArrangement = Arrangement.spacedBy(12.dp)
                ) {

                    CircularProgressIndicator()

                    Text("Analysing network traffic...")
                }
            }
        }

        errorMessage?.let { message ->

            Card(
                modifier = Modifier.fillMaxWidth()
            ) {

                Column(
                    modifier = Modifier.padding(16.dp)
                ) {

                    Text(
                        "Scanning Failed",
                        color = MaterialTheme.colorScheme.error,
                        style = MaterialTheme.typography.titleMedium
                    )

                    Spacer(Modifier.height(8.dp))

                    Text(message)
                }
            }
        }

        result?.let { summary ->

            Text(
                text = "Network Scan Results",
                style = MaterialTheme.typography.titleLarge
            )

            Card(
                modifier = Modifier.fillMaxWidth()
            ) {

                Column(
                    modifier = Modifier.padding(16.dp),
                    verticalArrangement = Arrangement.spacedBy(12.dp)
                ) {

                    Text(
                        "Total Network Flows",
                        style = MaterialTheme.typography.titleMedium
                    )

                    Text(
                        summary.totalRows.toString(),
                        style = MaterialTheme.typography.headlineMedium
                    )

                    Text(
                        "Benign Flows: ${summary.benignRows}",
                        color = MaterialTheme.colorScheme.primary
                    )

                    Text(
                        "Attack Flows: ${summary.attackRows}",
                        color = MaterialTheme.colorScheme.error
                    )

                    Text(
                        "Rows with missing or invalid values: " +
                                summary.rowsWithMissingValues
                    )
                }
            }

            Text(
                "Classification is performed locally using the trained Decision Tree model."
            )

            // Llama explanation
            Text(
                text = "AI Security Analysis",
                style = MaterialTheme.typography.titleLarge
            )

            Card(
                modifier = Modifier.fillMaxWidth()
            ) {

                Column(
                    modifier = Modifier.padding(16.dp),
                    verticalArrangement = Arrangement.spacedBy(10.dp)
                ) {

                    Text(
                        "Llama 3.2 1B Explanation",
                        fontWeight = FontWeight.Bold
                    )

                    if (isExplaining) {

                        Row(
                            horizontalArrangement = Arrangement.spacedBy(12.dp)
                        ) {

                            CircularProgressIndicator()

                            Text("Generating explanation...")
                        }

                    } else {

                        Text(explanation)
                    }
                }
            }
        }
    }
}
