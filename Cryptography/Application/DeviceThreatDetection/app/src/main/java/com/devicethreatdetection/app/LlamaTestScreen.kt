
package com.devicethreatdetection.app

import android.os.Handler
import android.os.Looper
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp

@Composable
fun LlamaTestScreen() {

    val context = LocalContext.current
    val assistant = remember { LlamaAssistant(context) }

    var isTesting by remember { mutableStateOf(false) }
    var result by remember { mutableStateOf("") }
    var error by remember { mutableStateOf("") }

    Card(
        modifier = Modifier.fillMaxWidth()
    ) {
        Column(
            modifier = Modifier.padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {

            Text(
                text = "LLAMA 3.2 AI TEST",
                fontSize = 20.sp,
                style = MaterialTheme.typography.titleLarge
            )

            Text(
                text = "Test local Llama inference using ExecuTorch on this device."
            )

            Button(
                onClick = {

                    isTesting = true
                    result = ""
                    error = ""

                    Thread {

                        try {

                            // Extract and load the local model.
                            assistant.initialize()

                            // Test text generation.
                            val response = assistant.explain(
                                """
                                <|begin_of_text|><|start_header_id|>user<|end_header_id|>

                                Explain in simple language what phishing is.
                                Give two safety recommendations.

                                <|eot_id|><|start_header_id|>assistant<|end_header_id|>
                                """.trimIndent()
                            )

                            Handler(Looper.getMainLooper()).post {
                                result = response
                                isTesting = false
                            }

                        } catch (e: Exception) {

                            Handler(Looper.getMainLooper()).post {
                                error = e.message
                                    ?: "Llama inference failed."
                                isTesting = false
                            }
                        }

                    }.start()
                },
                enabled = !isTesting,
                modifier = Modifier.fillMaxWidth()
            ) {
                Text(
                    if (isTesting) "Loading / Generating..."
                    else "Load Llama and Test"
                )
            }

            if (isTesting) {
                Text(
                    text = "Please wait. First-time model extraction and loading may take several minutes.",
                    color = MaterialTheme.colorScheme.primary
                )
            }

            if (result.isNotEmpty()) {

                Text(
                    text = "LLAMA RESPONSE",
                    style = MaterialTheme.typography.titleMedium
                )

                Text(text = result)
            }

            if (error.isNotEmpty()) {

                Text(
                    text = "ERROR",
                    color = MaterialTheme.colorScheme.error
                )

                Text(
                    text = error,
                    color = MaterialTheme.colorScheme.error
                )
            }
        }
    }
}
