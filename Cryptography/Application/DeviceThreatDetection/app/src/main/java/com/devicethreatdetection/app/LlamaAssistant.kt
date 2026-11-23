
package com.devicethreatdetection.app

import android.content.Context
import org.pytorch.executorch.extension.llm.LlmModule
import org.pytorch.executorch.extension.llm.LlmCallback
import org.pytorch.executorch.extension.llm.LlmGenerationConfig
import java.io.File
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit

class LlamaAssistant(private val context: Context) {

    companion object {
        private const val MODEL_FILE =
            "Llama-3.2-1B-Instruct-SpinQuant_INT4_EO8.pte"

        private const val TOKENIZER_FILE = "tokenizer.model"
    }

    private var module: LlmModule? = null

    private fun copyAsset(fileName: String): File {
        val directory = File(context.filesDir, "llama_model")

        if (!directory.exists()) {
            directory.mkdirs()
        }

        val destination = File(directory, fileName)

        if (!destination.exists()) {
            context.assets.open("models/$fileName").use { input ->
                destination.outputStream().use { output ->
                    input.copyTo(output)
                }
            }
        }

        return destination
    }

    fun initialize(): String {
        val modelFile = copyAsset(MODEL_FILE)
        val tokenizerFile = copyAsset(TOKENIZER_FILE)

        val llama = LlmModule(
            modelFile.absolutePath,
            tokenizerFile.absolutePath,
            0.2f
        )

        llama.load()
        module = llama

        return "Llama model loaded successfully."
    }

    fun explain(prompt: String): String {
        val llama = module
            ?: throw IllegalStateException(
                "Initialize Llama before generating text."
            )

        val response = StringBuilder()
        val latch = CountDownLatch(1)

        var generationError: String? = null

        val config = LlmGenerationConfig.create()
            .seqLen(512)
            .maxNewTokens(80)
            .temperature(0.2f)
            .echo(false)
            .build()

        val callback = object : LlmCallback {

            override fun onResult(token: String) {
                response.append(token)
            }

            override fun onStats(statsJson: String) {
                latch.countDown()
            }

            override fun onError(errorCode: Int, message: String) {
                generationError = "Error $errorCode: $message"
                latch.countDown()
            }
        }

        llama.resetContext()
        llama.generate(prompt, config, callback)

        val completed = latch.await(180, TimeUnit.SECONDS)

        if (!completed) {
            throw IllegalStateException(
                "Llama generation timed out."
            )
        }

        generationError?.let {
            throw IllegalStateException(it)
        }

        val result = response.toString().trim()

        if (result.isEmpty()) {
            throw IllegalStateException(
                "Llama returned an empty response."
            )
        }

        return result
    }
}
