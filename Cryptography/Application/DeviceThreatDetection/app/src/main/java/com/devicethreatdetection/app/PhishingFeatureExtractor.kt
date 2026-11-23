
package com.devicethreatdetection.app

import android.content.Context
import java.net.URI
import kotlin.math.max

class PhishingFeatureExtractor(context: Context) {

    private val brandSet = HashSet<String>()
    private val brandIndex = HashMap<Char, MutableList<String>>()
    private val brandLengths = sortedSetOf<Int>()

    private val suffixRules = HashSet<String>()
    private val wildcardRules = HashSet<String>()
    private val exceptionRules = HashSet<String>()

    init {
        loadBrandLibrary(context)
        loadPublicSuffixList(context)
    }

    private fun loadBrandLibrary(context: Context) {
        context.assets.open("brand_library.txt")
            .bufferedReader()
            .useLines { lines ->
                lines.forEach { line ->
                    val brand = normalizeBrand(line)

                    if (brand.isNotEmpty() && brandSet.add(brand)) {
                        brandIndex
                            .getOrPut(brand[0]) { mutableListOf() }
                            .add(brand)

                        if (brand.length >= 4) {
                            brandLengths.add(brand.length)
                        }
                    }
                }
            }
    }

    private fun loadPublicSuffixList(context: Context) {
        context.assets.open("public_suffix_list.dat")
            .bufferedReader()
            .useLines { lines ->
                lines.forEach { original ->
                    val rule = original.trim()

                    if (rule.isEmpty() || rule.startsWith("//")) {
                        return@forEach
                    }

                    when {
                        rule.startsWith("!") ->
                            exceptionRules.add(rule.substring(1))

                        rule.startsWith("*.") ->
                            wildcardRules.add(rule.substring(2))

                        else ->
                            suffixRules.add(rule)
                    }
                }
            }
    }

    // --------------------------------------------------------
    // NORMALIZATION
    // --------------------------------------------------------

    private fun normalizeBrand(value: String): String {
        return value.lowercase()
            .replace(Regex("[^a-z0-9]"), "")
    }

    private fun addScheme(url: String): String {
        val value = url.trim()

        return if (
            Regex("^[a-zA-Z][a-zA-Z0-9+.-]*://")
                .containsMatchIn(value)
        ) {
            value
        } else {
            "https://$value"
        }
    }

    private fun codePointLength(value: String): Int {
        return value.codePointCount(0, value.length)
    }

    // --------------------------------------------------------
    // HOSTNAME EXTRACTION
    // --------------------------------------------------------

    private fun extractHostname(url: String): String {
        val normalized = addScheme(url)

        return try {
            val uri = URI(normalized)

            var host = uri.host

            if (host.isNullOrEmpty()) {
                var authority = uri.rawAuthority ?: ""

                if (authority.contains("@")) {
                    authority = authority.substringAfterLast("@")
                }

                if (authority.startsWith("[")) {
                    host = authority.substringBefore("]") + "]"
                } else {
                    host = authority.substringBefore(":")
                }
            }

            host.lowercase().trim('.')

        } catch (e: Exception) {
            val withoutScheme =
                normalized.substringAfter("://", normalized)

            withoutScheme
                .substringBefore("/")
                .substringBefore("?")
                .substringBefore("#")
                .substringAfterLast("@")
                .substringBefore(":")
                .lowercase()
                .trim('.')
        }
    }

    // --------------------------------------------------------
    // REGISTERED DOMAIN EXTRACTION
    // --------------------------------------------------------

    private fun extractDomainLabel(hostname: String): String {

        if (hostname.isEmpty()) return ""

        val labels = hostname.split(".")

        if (labels.size < 2) {
            return labels.lastOrNull() ?: ""
        }

        var publicSuffixLength = 1

        for (i in labels.indices) {
            val candidate = labels.drop(i).joinToString(".")

            if (exceptionRules.contains(candidate)) {
                publicSuffixLength = labels.size - i - 1
                break
            }
        }

        if (publicSuffixLength == 1) {
            for (i in labels.indices) {
                val candidate = labels.drop(i).joinToString(".")

                if (suffixRules.contains(candidate)) {
                    publicSuffixLength = max(
                        publicSuffixLength,
                        labels.size - i
                    )
                }

                if (i + 1 < labels.size) {
                    val wildcardCandidate =
                        labels.drop(i + 1).joinToString(".")

                    if (wildcardRules.contains(wildcardCandidate)) {
                        publicSuffixLength = max(
                            publicSuffixLength,
                            labels.size - i
                        )
                    }
                }
            }
        }

        val domainIndex = labels.size - publicSuffixLength - 1

        return if (domainIndex >= 0) {
            labels[domainIndex]
        } else {
            labels.first()
        }
    }

    // --------------------------------------------------------
    // BRAND SIMILARITY
    // --------------------------------------------------------

    private fun indelRatio(a: String, b: String): Double {

        if (a.isEmpty() && b.isEmpty()) return 1.0
        if (a.isEmpty() || b.isEmpty()) return 0.0

        val first = a.codePoints().toArray()
        val second = b.codePoints().toArray()

        var previous = IntArray(second.size + 1)
        var current = IntArray(second.size + 1)

        for (i in first.indices) {
            for (j in second.indices) {
                current[j + 1] = if (first[i] == second[j]) {
                    previous[j] + 1
                } else {
                    max(previous[j + 1], current[j])
                }
            }

            val temp = previous
            previous = current
            current = temp

            current.fill(0)
        }

        val lcs = previous[second.size]

        return (2.0 * lcs) / (first.size + second.size)
    }

    private fun calculateBrandSimilarity(domain: String): Double {

        val normalized = normalizeBrand(domain)

        if (normalized.isEmpty()) return 0.0

        if (brandSet.contains(normalized)) return 1.0

        val candidates = brandIndex[normalized[0]] ?: return 0.0

        var bestScore = 0.0

        for (brand in candidates) {

            val denominator = normalized.length + brand.length

            if (denominator == 0) continue

            val maximumPossible =
                (2.0 * minOf(normalized.length, brand.length)) /
                        denominator

            if (maximumPossible <= bestScore) continue

            val score = indelRatio(normalized, brand)

            if (score > bestScore) {
                bestScore = score
            }

            if (bestScore >= 1.0) {
                return 1.0
            }
        }

        return bestScore
    }

    // --------------------------------------------------------
    // BRAND SUBSTRING MATCH
    // --------------------------------------------------------

    private fun calculateBrandSubstringMatch(domain: String): Int {

        val normalized = normalizeBrand(domain)

        if (normalized.isEmpty()) return 0

        val domainLength = normalized.length

        for (start in 0 until domainLength) {
            for (brandLength in brandLengths) {

                val end = start + brandLength

                if (end > domainLength) break

                val candidate = normalized.substring(start, end)

                if (brandSet.contains(candidate)) {
                    return 1
                }
            }
        }

        return 0
    }

    // --------------------------------------------------------
    // EXTRACT MODEL B FEATURES
    // HOSTNAME-ONLY MODE
    // --------------------------------------------------------

    fun extractFeatures(url: String): DoubleArray {

        require(url.isNotBlank()) {
            "Please enter a URL."
        }

        val hostname = extractHostname(url)

        require(hostname.isNotBlank()) {
            "Unable to extract a valid hostname."
        }

        // Extract registered domain for brand matching.
        val domainName = extractDomainLabel(hostname)

        val normalizedDomain = normalizeBrand(domainName)

        val brandSimilarity =
            calculateBrandSimilarity(normalizedDomain)

        val exactBrandMatch = if (
            normalizedDomain.isNotEmpty() &&
            brandSet.contains(normalizedDomain)
        ) {
            1
        } else {
            0
        }

        val brandSubstringMatch =
            calculateBrandSubstringMatch(normalizedDomain)

        // All URL-character features now use the hostname only.
        // The URL path and query parameters are excluded.

        return doubleArrayOf(

            // 0. hostname_length
            codePointLength(hostname).toDouble(),

            // 1. dot_count
            hostname.count { it == '.' }.toDouble(),

            // 2. hyphen_count
            hostname.count { it == '-' }.toDouble(),

            // 3. at_count
            hostname.count { it == '@' }.toDouble(),

            // 4. digit_count
            hostname.codePoints()
                .filter { Character.isDigit(it) }
                .count()
                .toDouble(),

            // 5. subdomain_count
            max(0, hostname.split(".").size - 2).toDouble(),

            // 6. path_length
            0.0,

            // 7. brand_similarity
            brandSimilarity,

            // 8. exact_brand_match
            exactBrandMatch.toDouble(),

            // 9. brand_substring_match
            brandSubstringMatch.toDouble()
        )
    }
}
