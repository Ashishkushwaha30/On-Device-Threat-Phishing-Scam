
package com.devicethreatdetection.app

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.imePadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.devicethreatdetection.app.ui.theme.DeviceThreatDetectionTheme
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

// --------------------------------------------------
// SCAN HISTORY DATA
// --------------------------------------------------

data class ScanHistoryItem(
    val module: String,
    val input: String = "",
    val result: String,
    val scannedCount: Int,
    val threatCount: Int,
    val timestamp: String
)


// --------------------------------------------------
// MAIN ACTIVITY
// --------------------------------------------------

class MainActivity : ComponentActivity() {

    // Shared scan history
    val scanHistory = mutableStateListOf<ScanHistoryItem>()

    private lateinit var historyStorage: ScanHistoryStorage

    // Record a completed scan
    fun recordScan(
        module: String,
        result: String,
        scannedCount: Int,
        threatCount: Int,
        input: String = ""
    ) {

        val timestamp = SimpleDateFormat(
            "dd MMM, hh:mm a",
            Locale.getDefault()
        ).format(Date())

        scanHistory.add(
            0,
            ScanHistoryItem(
                module = module,
                input = input,
                result = result,
                scannedCount = scannedCount,
                threatCount = threatCount,
                timestamp = timestamp
            )
        )

        // Keep only the latest 50 records
        while (scanHistory.size > 50) {
            scanHistory.removeAt(scanHistory.lastIndex)
        }

        // Save history permanently on the device
        historyStorage.saveHistory(scanHistory.toList())
    }

    // Phishing URL scanner
    private val phishingScanner by lazy(
        LazyThreadSafetyMode.SYNCHRONIZED
    ) {
        PhishingScanner(this)
    }

    // SMS spam scanner
    private val smsSpamScanner by lazy(
        LazyThreadSafetyMode.SYNCHRONIZED
    ) {
        SmsSpamScanner(this)
    }

    // Network intrusion scanner
    private val networkIntrusionScanner by lazy(
        LazyThreadSafetyMode.SYNCHRONIZED
    ) {
        NetworkIntrusionScanner(applicationContext)
    }

    // Activity-level Llama instance
    @Volatile
    private var llamaAssistant: LlamaAssistant? = null

    // Observable Llama loading status
    var llamaLoading by mutableStateOf(false)
        private set

    var llamaStatus by mutableStateOf("Llama model is not loaded.")
        private set

    // Phishing scanning
    fun scanUrlWithDetails(url: String): PhishingScanResult {
        return phishingScanner.scanUrlWithDetails(url)
    }

    fun scanPhishingUrl(url: String): String {
        return phishingScanner.scanUrl(url)
    }

    // SMS scanning
    fun scanSmsMessage(message: String): SmsScanResult {
        return smsSpamScanner.scanSms(message)
    }

    // Network scanning
    fun obtainNetworkIntrusionScanner(): NetworkIntrusionScanner {
        return networkIntrusionScanner
    }

    // Llama initialization
    @Synchronized
    fun initializeLlamaAssistant(): String {

        if (llamaAssistant != null) {
            return "Llama model is already loaded."
        }

        val assistant = LlamaAssistant(applicationContext)
        val message = assistant.initialize()

        llamaAssistant = assistant

        return message
    }

    // Automatically load Llama in the background
    @Synchronized
    fun startLlamaAutoLoad() {

        // Avoid loading the model more than once
        if (llamaAssistant != null || llamaLoading) {
            return
        }

        llamaLoading = true
        llamaStatus = "Loading Llama model. Please wait..."

        Thread {

            try {

                val message = initializeLlamaAssistant()

                runOnUiThread {
                    llamaStatus = message
                    llamaLoading = false
                }

            } catch (e: Exception) {

                runOnUiThread {
                    llamaStatus =
                        "Model loading failed: ${e.message ?: "Unknown error"}"
                    llamaLoading = false
                }
            }

        }.start()
    }

    fun isLlamaLoaded(): Boolean {
        return llamaAssistant != null
    }

    @Synchronized
    fun generateLlamaResponse(prompt: String): String {

        val assistant = llamaAssistant
            ?: throw IllegalStateException("Please load Llama first.")

        return assistant.explain(prompt)
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Initialize permanent history storage
        historyStorage = ScanHistoryStorage(applicationContext)

        // Restore previously saved scan history
        scanHistory.addAll(historyStorage.loadHistory())

        setContent {
            DeviceThreatDetectionTheme {
                DeviceThreatDetectionApp(this)
            }
        }

        // Start Llama loading automatically
        startLlamaAutoLoad()
    }
}


// --------------------------------------------------
// MAIN DASHBOARD NAVIGATION
// --------------------------------------------------

@Composable
fun DeviceThreatDetectionApp(activity: MainActivity) {

    var selectedModule by remember {
        mutableStateOf("Dashboard")
    }

    Surface(
        modifier = Modifier.fillMaxSize(),
        color = MaterialTheme.colorScheme.background
    ) {

        when (selectedModule) {

            "Dashboard" -> {

                DashboardScreen(
                    activity = activity,
                    onModuleSelected = { module ->
                        selectedModule = module
                    }
                )
            }

            "Network Detection" -> {

                NetworkDetectionScreen(
                    activity = activity,
                    onBack = {
                        selectedModule = "Dashboard"
                    }
                )
            }

            "Phishing URL Scanner" -> {

                PhishingScannerScreen(
                    activity = activity,
                    onBack = {
                        selectedModule = "Dashboard"
                    }
                )
            }

            "SMS Security" -> {

                SmsSecurityScreen(
                    activity = activity,
                    onBack = {
                        selectedModule = "Dashboard"
                    }
                )
            }

            "On-device AI Assistant" -> {

                LlamaTestScreen(
                    activity = activity,
                    onBack = {
                        selectedModule = "Dashboard"
                    }
                )
            }
        }
    }
}


// --------------------------------------------------
// DASHBOARD MODULE CARD
// --------------------------------------------------

@Composable
fun ModuleCard(
    title: String,
    description: String,
    onClick: () -> Unit
) {

    Card(
        modifier = Modifier.fillMaxWidth(),
        elevation = CardDefaults.cardElevation(
            defaultElevation = 4.dp
        ),
        onClick = onClick
    ) {

        Column(
            modifier = Modifier.padding(18.dp)
        ) {

            Text(
                text = title,
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.Bold
            )

            Spacer(modifier = Modifier.height(6.dp))

            Text(
                text = description,
                style = MaterialTheme.typography.bodyMedium
            )
        }
    }
}


// --------------------------------------------------
// PHISHING URL SCANNER
// --------------------------------------------------

@Composable
fun PhishingScannerScreen(
    activity: MainActivity,
    onBack: () -> Unit
) {

    var inputUrl by remember { mutableStateOf("") }
    var result by remember { mutableStateOf("") }
    var explanation by remember { mutableStateOf("") }
    var isScanning by remember { mutableStateOf(false) }
    var isExplaining by remember { mutableStateOf(false) }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .imePadding()
            .padding(20.dp),
        verticalArrangement = Arrangement.spacedBy(14.dp)
    ) {

        Button(onClick = onBack) {
            Text("Back")
        }

        Text(
            text = "Phishing URL Scanner",
            style = MaterialTheme.typography.headlineSmall,
            fontWeight = FontWeight.Bold
        )

        Text(
            "Enter a URL to analyse it using the local Decision Tree model."
        )

        OutlinedTextField(
            value = inputUrl,
            onValueChange = { inputUrl = it },
            modifier = Modifier.fillMaxWidth(),
            label = { Text("Enter URL") },
            placeholder = { Text("https://example.com") },
            singleLine = true
        )

        Button(
            onClick = {

                if (inputUrl.isNotBlank() &&
                    !isScanning && !isExplaining
                ) {

                    val urlToScan = inputUrl.trim()

                    isScanning = true
                    result = "Analysing URL..."
                    explanation = ""

                    Thread {

                        try {

                            val scan =
                                activity.scanUrlWithDetails(urlToScan)

                            activity.runOnUiThread {

                                result = scan.prediction
                                isScanning = false

                                // Record successful URL scan
                                activity.recordScan(
                                    module = "URL",
                                    input = urlToScan,
                                    result = scan.prediction,
                                    scannedCount = 1,
                                    threatCount = if (
                                        scan.prediction.equals(
                                            "Phishing",
                                            ignoreCase = true
                                        )
                                    ) 1 else 0
                                )

                                if (activity.isLlamaLoaded()) {

                                    isExplaining = true
                                    explanation =
                                        "Generating AI explanation..."

                                    Thread {

                                        try {

                                            val isPhishing =
                                                scan.prediction.equals(
                                                    "Phishing",
                                                    ignoreCase = true
                                                )

                                            val prompt = if (isPhishing) {

                                                """
                                                <|begin_of_text|><|start_header_id|>system<|end_header_id|>

                                                You explain an existing URL security classification.

                                                IMPORTANT: The Decision Tree prediction is final. Never change or contradict it.

                                                The scanner classified this URL as PHISHING.

                                                Explain this result in simple English. Do not invent website behaviour, stolen credentials, malware, fake login pages, or any other unsupported facts.

                                                Advise the user not to open the link or enter passwords, OTPs, or banking details. Suggest visiting the official website directly.

                                                Write 3 to 4 short sentences in one paragraph. No headings or bullet points.

                                                <|eot_id|><|start_header_id|>user<|end_header_id|>

                                                URL: ${urlToScan.take(200)}
                                                Final classification: PHISHING.

                                                Explain this exact classification. Do not change it.

                                                <|eot_id|><|start_header_id|>assistant<|end_header_id|>
                                                """.trimIndent()

                                            } else {

                                                """
                                                <|begin_of_text|><|start_header_id|>system<|end_header_id|>

                                                You explain an existing URL security classification.

                                                IMPORTANT: The Decision Tree prediction is final. Never change or contradict it.

                                                The scanner classified this URL as LEGITIMATE.

                                                Explain that the URL passed this phishing check. Do not call it phishing, malicious, suspicious, or a scam.

                                                Do not invent threats or website behaviour. Do not claim that you independently verified the website.

                                                Briefly mention that passing this check does not guarantee complete safety. Recommend checking the address before sharing sensitive information.

                                                Write 3 to 4 short sentences in one paragraph. No headings or bullet points.

                                                <|eot_id|><|start_header_id|>user<|end_header_id|>

                                                URL: ${urlToScan.take(200)}
                                                Final classification: LEGITIMATE.

                                                Explain this exact classification. Do not change it.

                                                <|eot_id|><|start_header_id|>assistant<|end_header_id|>
                                                """.trimIndent()
                                            }

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
                                        "Load Llama from the AI Assistant module to get an explanation."
                                }
                            }

                        } catch (e: Exception) {

                            activity.runOnUiThread {
                                result = "Error: ${e.message}"
                                isScanning = false
                            }
                        }

                    }.start()
                }
            },
            enabled = inputUrl.isNotBlank() &&
                    !isScanning && !isExplaining
        ) {

            Text(
                when {
                    isScanning -> "Scanning..."
                    isExplaining -> "Generating Explanation..."
                    else -> "Scan URL"
                }
            )
        }

        if (result.isNotBlank()) {

            Card(modifier = Modifier.fillMaxWidth()) {

                Column(modifier = Modifier.padding(16.dp)) {

                    Text(
                        "Decision Tree Prediction",
                        fontWeight = FontWeight.Bold
                    )

                    Spacer(modifier = Modifier.height(8.dp))

                    Text(
                        result,
                        style = MaterialTheme.typography.titleLarge
                    )
                }
            }
        }

        if (explanation.isNotBlank()) {

            Card(modifier = Modifier.fillMaxWidth()) {

                Column(modifier = Modifier.padding(16.dp)) {

                    Text(
                        "Llama AI Explanation",
                        fontWeight = FontWeight.Bold
                    )

                    Spacer(modifier = Modifier.height(8.dp))

                    Text(explanation)
                }
            }
        }
    }
}


// --------------------------------------------------
// SMS SPAM DETECTION
// --------------------------------------------------

@Composable
fun SmsSecurityScreen(
    activity: MainActivity,
    onBack: () -> Unit
) {

    var smsText by remember { mutableStateOf("") }
    var prediction by remember { mutableStateOf("") }
    var explanation by remember { mutableStateOf("") }
    var isScanning by remember { mutableStateOf(false) }
    var isExplaining by remember { mutableStateOf(false) }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .imePadding()
            .padding(20.dp),
        verticalArrangement = Arrangement.spacedBy(14.dp)
    ) {

        Button(onClick = onBack) {
            Text("Back")
        }

        Text(
            text = "SMS Security",
            style = MaterialTheme.typography.headlineSmall,
            fontWeight = FontWeight.Bold
        )

        Text(
            "Enter or paste an SMS message to check whether it is spam or legitimate."
        )

        OutlinedTextField(
            value = smsText,
            onValueChange = { smsText = it },
            modifier = Modifier.fillMaxWidth(),
            label = { Text("Enter SMS message") },
            placeholder = { Text("Paste your SMS message here...") },
            minLines = 4,
            maxLines = 7
        )

        Button(
            onClick = {

                if (smsText.isNotBlank() &&
                    !isScanning && !isExplaining
                ) {

                    val messageToScan = smsText.trim()

                    isScanning = true
                    prediction = "Analysing SMS..."
                    explanation = ""

                    Thread {

                        try {

                            val scan =
                                activity.scanSmsMessage(messageToScan)

                            activity.runOnUiThread {

                                prediction = scan.prediction
                                isScanning = false

                                // Record successful SMS scan
                                activity.recordScan(
                                    module = "SMS",
                                    input = messageToScan,
                                    result = scan.prediction,
                                    scannedCount = 1,
                                    threatCount = if (
                                        scan.prediction.equals(
                                            "Spam",
                                            ignoreCase = true
                                        )
                                    ) 1 else 0
                                )

                                if (activity.isLlamaLoaded()) {

                                    isExplaining = true
                                    explanation =
                                        "Generating AI explanation..."

                                    Thread {

                                        try {

                                            val safeMessage =
                                                messageToScan.take(250)

                                            val isSpam =
                                                scan.prediction.equals(
                                                    "Spam",
                                                    ignoreCase = true
                                                )

                                            val prompt = if (isSpam) {

                                                """
                                                <|begin_of_text|><|start_header_id|>system<|end_header_id|>

                                                You explain an SMS classification that has already been made.

                                                IMPORTANT: The Decision Tree prediction is FINAL. Never change or contradict it.

                                                This SMS was classified as SPAM.

                                                Explain that the scanner identified this message as spam. Mention only characteristics that actually appear in the supplied SMS.

                                                Do not invent a sender identity, link, bank request, payment request, or OTP request if it is not present.

                                                Advise the user not to follow suspicious instructions or share sensitive information. Recommend verifying unexpected requests through an official channel.

                                                Write 3 to 4 short sentences in one paragraph. No headings or bullet points.

                                                <|eot_id|><|start_header_id|>user<|end_header_id|>

                                                SMS:
                                                $safeMessage

                                                Final classification: SPAM.

                                                Explain this exact classification. Do not change it.

                                                <|eot_id|><|start_header_id|>assistant<|end_header_id|>
                                                """.trimIndent()

                                            } else {

                                                """
                                                <|begin_of_text|><|start_header_id|>system<|end_header_id|>

                                                You explain an SMS classification that has already been made.

                                                IMPORTANT: The Decision Tree prediction is FINAL. Never change or contradict it.

                                                This SMS was classified as HAM (LEGITIMATE).

                                                Explain that the scanner classified the message as a normal or legitimate message.

                                                Do not call it spam, phishing, fraudulent, malicious, or suspicious. Do not invent a threat, scam, request for credentials, or intention to steal information.

                                                Mention only what the supplied SMS actually says. You may briefly state that the classification is not a guarantee of the sender's identity, but do not turn this into an unsupported warning about this particular message.

                                                Write 3 to 4 short sentences in one paragraph. No headings or bullet points.

                                                <|eot_id|><|start_header_id|>user<|end_header_id|>

                                                SMS:
                                                $safeMessage

                                                Final classification: HAM (LEGITIMATE).

                                                Explain this exact classification. Do not change it.

                                                <|eot_id|><|start_header_id|>assistant<|end_header_id|>
                                                """.trimIndent()
                                            }

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
                                        "Load Llama from the AI Assistant module to get an explanation."
                                }
                            }

                        } catch (e: Exception) {

                            activity.runOnUiThread {
                                prediction =
                                    "Scanning failed: ${e.message}"
                                isScanning = false
                            }
                        }

                    }.start()
                }
            },
            enabled = smsText.isNotBlank() &&
                    !isScanning && !isExplaining
        ) {

            Text(
                when {
                    isScanning -> "Scanning SMS..."
                    isExplaining -> "Generating Explanation..."
                    else -> "Scan SMS"
                }
            )
        }

        if (prediction.isNotBlank()) {

            Card(modifier = Modifier.fillMaxWidth()) {

                Column(modifier = Modifier.padding(16.dp)) {

                    Text(
                        "Decision Tree Prediction",
                        fontWeight = FontWeight.Bold
                    )

                    Spacer(modifier = Modifier.height(8.dp))

                    Text(
                        prediction,
                        style = MaterialTheme.typography.titleLarge,
                        fontWeight = FontWeight.Bold
                    )
                }
            }
        }

        if (explanation.isNotBlank()) {

            Card(modifier = Modifier.fillMaxWidth()) {

                Column(modifier = Modifier.padding(16.dp)) {

                    Text(
                        "Llama AI Explanation",
                        fontWeight = FontWeight.Bold
                    )

                    Spacer(modifier = Modifier.height(8.dp))

                    Text(explanation)
                }
            }
        }
    }
}


// --------------------------------------------------
// LLAMA AI ASSISTANT SCREEN
// --------------------------------------------------

@Composable
fun LlamaTestScreen(
    activity: MainActivity,
    onBack: () -> Unit
) {

    var isLoaded by remember {
        mutableStateOf(activity.isLlamaLoaded())
    }

    var status by remember {
        mutableStateOf(
            if (activity.isLlamaLoaded()) {
                "Llama model is already loaded."
            } else {
                "Llama model is not loaded."
            }
        )
    }

    var response by remember { mutableStateOf("") }
    var isLoading by remember { mutableStateOf(false) }
    var isGenerating by remember { mutableStateOf(false) }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .imePadding()
            .padding(20.dp),
        verticalArrangement = Arrangement.spacedBy(14.dp)
    ) {

        Button(onClick = onBack) {
            Text("Back")
        }

        Text(
            text = "On-device AI Assistant",
            style = MaterialTheme.typography.headlineSmall,
            fontWeight = FontWeight.Bold
        )

        Text(
            "Llama 3.2 1B Instruct running locally through ExecuTorch."
        )

        Card(modifier = Modifier.fillMaxWidth()) {

            Text(
                status,
                modifier = Modifier.padding(16.dp)
            )
        }

        Button(
            onClick = {

                if (!isLoading && !isLoaded) {

                    isLoading = true
                    status = "Loading Llama model. Please wait..."
                    response = ""

                    Thread {

                        try {

                            val message =
                                activity.initializeLlamaAssistant()

                            activity.runOnUiThread {
                                status = message
                                isLoaded = true
                                isLoading = false
                            }

                        } catch (e: Exception) {

                            activity.runOnUiThread {
                                status =
                                    "Model loading failed: ${e.message}"
                                isLoading = false
                            }
                        }

                    }.start()
                }
            },
            enabled = !isLoading && !isLoaded
        ) {

            Text(
                when {
                    isLoading -> "Loading Model..."
                    isLoaded -> "Model Loaded"
                    else -> "Load Llama Model"
                }
            )
        }

        Button(
            onClick = {

                if (isLoaded && !isGenerating) {

                    isGenerating = true
                    response = "Generating response..."

                    Thread {

                        try {

                            val prompt = """
                                <|begin_of_text|><|start_header_id|>system<|end_header_id|>

                                You are a friendly cybersecurity assistant helping ordinary smartphone users.

                                Explain why people should be careful while browsing the internet.

                                Use simple everyday English.
                                Write 5 short complete sentences in one paragraph.
                                Do not use headings, bullet points, technical terminology or Markdown.

                                <|eot_id|><|start_header_id|>user<|end_header_id|>

                                Explain five important ways to stay safe from online scams and phishing websites.

                                <|eot_id|><|start_header_id|>assistant<|end_header_id|>
                            """.trimIndent()

                            val generatedText =
                                activity.generateLlamaResponse(prompt)

                            activity.runOnUiThread {
                                response = generatedText
                                isGenerating = false
                            }

                        } catch (e: Exception) {

                            activity.runOnUiThread {
                                response =
                                    "Generation failed: ${e.message}"
                                isGenerating = false
                            }
                        }

                    }.start()
                }
            },
            enabled = isLoaded && !isGenerating
        ) {

            Text(
                if (isGenerating) {
                    "Generating..."
                } else {
                    "Generate Test Response"
                }
            )
        }

        if (response.isNotBlank()) {

            Card(modifier = Modifier.fillMaxWidth()) {

                Text(
                    response,
                    modifier = Modifier.padding(16.dp)
                )
            }
        }
    }
}
