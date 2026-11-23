
package com.devicethreatdetection.app

import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp

@Composable
fun DashboardScreen(
    activity: MainActivity,
    onModuleSelected: (String) -> Unit
) {
    val history = activity.scanHistory

    var selectedHistoryItem by remember {
        mutableStateOf<ScanHistoryItem?>(null)
    }

    // Separate history for each module
    val urlHistory = history.filter { it.module == "URL" }
    val smsHistory = history.filter { it.module == "SMS" }
    val networkHistory = history.filter { it.module == "Network" }

    // URL statistics
    val urlScans = urlHistory.size
    val phishingThreats = urlHistory.sumOf { it.threatCount }

    // SMS statistics
    val smsScans = smsHistory.size
    val spamThreats = smsHistory.sumOf { it.threatCount }

    // Network statistics
    val totalNetworkFlows = networkHistory.sumOf { it.scannedCount }
    val networkThreats = networkHistory.sumOf { it.threatCount }

    // Overall statistics
    val totalThreats =
        phishingThreats + spamThreats + networkThreats

    val totalScanSessions =
        urlScans + smsScans + networkHistory.size

    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(20.dp),
        verticalArrangement = Arrangement.spacedBy(18.dp)
    ) {

        // APP HEADER

        Text(
            text = "DEVICE THREAT",
            style = MaterialTheme.typography.headlineMedium,
            fontWeight = FontWeight.ExtraBold,
            color = MaterialTheme.colorScheme.primary
        )

        Text(
            text = "Detection Center",
            style = MaterialTheme.typography.titleLarge,
            fontWeight = FontWeight.Bold
        )

        Text(
            text = "Your personal on-device security dashboard",
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )

        // SECURITY OVERVIEW

        Card(
            modifier = Modifier.fillMaxWidth(),
            colors = CardDefaults.cardColors(
                containerColor = MaterialTheme.colorScheme.primaryContainer
            )
        ) {
            Column(
                modifier = Modifier.padding(18.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                Text(
                    text = "SECURITY OVERVIEW",
                    style = MaterialTheme.typography.labelLarge,
                    fontWeight = FontWeight.Bold,
                    color = MaterialTheme.colorScheme.onPrimaryContainer
                )

                Text(
                    text = when {
                        totalScanSessions == 0 ->
                            "Your device security dashboard is ready"

                        totalThreats > 0 ->
                            "Potential threats detected in your scans"

                        else ->
                            "No threats flagged in recorded scans"
                    },
                    style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.Bold,
                    color = MaterialTheme.colorScheme.onPrimaryContainer
                )

                Text(
                    text = "Completed scan sessions: $totalScanSessions",
                    color = MaterialTheme.colorScheme.onPrimaryContainer
                )

                Text(
                    text = "Potential threats flagged: $totalThreats",
                    color = MaterialTheme.colorScheme.onPrimaryContainer
                )

                Text(
                    text = "Processing: On-device",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onPrimaryContainer
                )
            }
        }

        // SECURITY MODULES

        Text(
            text = "Security Modules",
            style = MaterialTheme.typography.titleLarge,
            fontWeight = FontWeight.Bold
        )

        // First row
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            DashboardModuleCard(
                title = "Network Detection",
                description = "Detect suspicious network activity",
                symbol = "🌐",
                containerColor = MaterialTheme.colorScheme.primaryContainer,
                modifier = Modifier.weight(1f)
            ) {
                onModuleSelected("Network Detection")
            }

            DashboardModuleCard(
                title = "Phishing URL Scanner",
                description = "Identify potentially harmful URLs",
                symbol = "🔗",
                containerColor = MaterialTheme.colorScheme.secondaryContainer,
                modifier = Modifier.weight(1f)
            ) {
                onModuleSelected("Phishing URL Scanner")
            }
        }

        // Second row
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            DashboardModuleCard(
                title = "SMS Security",
                description = "Detect spam and fraudulent messages",
                symbol = "💬",
                containerColor = MaterialTheme.colorScheme.tertiaryContainer,
                modifier = Modifier.weight(1f)
            ) {
                onModuleSelected("SMS Security")
            }

            DashboardModuleCard(
                title = "On-device AI Assistant",
                description = "Get explanations using local Llama",
                symbol = "🤖",
                containerColor = MaterialTheme.colorScheme.errorContainer,
                modifier = Modifier.weight(1f)
            ) {
                onModuleSelected("On-device AI Assistant")
            }
        }

        // DETECTION STATISTICS

        Text(
            text = "Detection Statistics",
            style = MaterialTheme.typography.titleLarge,
            fontWeight = FontWeight.Bold
        )

        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(10.dp)
        ) {
            StatisticCard(
                title = "URL Scans",
                value = urlScans.toString(),
                containerColor = MaterialTheme.colorScheme.primaryContainer,
                modifier = Modifier.weight(1f)
            )

            StatisticCard(
                title = "Phishing",
                value = phishingThreats.toString(),
                containerColor = MaterialTheme.colorScheme.errorContainer,
                modifier = Modifier.weight(1f)
            )
        }

        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(10.dp)
        ) {
            StatisticCard(
                title = "SMS Scans",
                value = smsScans.toString(),
                containerColor = MaterialTheme.colorScheme.secondaryContainer,
                modifier = Modifier.weight(1f)
            )

            StatisticCard(
                title = "Spam",
                value = spamThreats.toString(),
                containerColor = MaterialTheme.colorScheme.errorContainer,
                modifier = Modifier.weight(1f)
            )
        }

        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(10.dp)
        ) {
            StatisticCard(
                title = "Network Flows",
                value = totalNetworkFlows.toString(),
                containerColor = MaterialTheme.colorScheme.tertiaryContainer,
                modifier = Modifier.weight(1f)
            )

            StatisticCard(
                title = "Attack Flows",
                value = networkThreats.toString(),
                containerColor = MaterialTheme.colorScheme.errorContainer,
                modifier = Modifier.weight(1f)
            )
        }

        // SCAN HISTORY (AT THE BOTTOM)

        Text(
            text = "Scan History",
            style = MaterialTheme.typography.titleLarge,
            fontWeight = FontWeight.Bold
        )

        Card(
            modifier = Modifier.fillMaxWidth(),
            colors = CardDefaults.cardColors(
                containerColor = MaterialTheme.colorScheme.surfaceContainer
            )
        ) {
            Column(
                modifier = Modifier.padding(14.dp),
                verticalArrangement = Arrangement.spacedBy(10.dp)
            ) {
                if (history.isEmpty()) {
                    Text(
                        text = "No scans recorded yet.",
                        style = MaterialTheme.typography.titleMedium,
                        fontWeight = FontWeight.Medium
                    )

                    Text(
                        text = "Start using a security module to see your scan history here.",
                        style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                } else {
                    Text(
                        text = "Tap any entry to view its input and prediction.",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )

                    history.take(10).forEach { item ->

                        Card(
                            modifier = Modifier
                                .fillMaxWidth()
                                .clickable {
                                    selectedHistoryItem = item
                                },
                            colors = CardDefaults.cardColors(
                                containerColor = MaterialTheme.colorScheme.surfaceVariant
                            )
                        ) {
                            Column(
                                modifier = Modifier.padding(14.dp),
                                verticalArrangement = Arrangement.spacedBy(5.dp)
                            ) {
                                Text(
                                    text = when (item.module) {
                                        "URL" -> "🔗 Phishing URL Scanner"
                                        "SMS" -> "💬 SMS Security"
                                        "Network" -> "🌐 Network Detection"
                                        else -> item.module
                                    },
                                    fontWeight = FontWeight.Bold,
                                    style = MaterialTheme.typography.titleSmall
                                )

                                Text(
                                    text = item.result,
                                    style = MaterialTheme.typography.bodyMedium
                                )

                                Text(
                                    text = "${item.scannedCount} item(s) scanned · ${item.threatCount} threat(s) flagged",
                                    style = MaterialTheme.typography.bodySmall,
                                    color = MaterialTheme.colorScheme.onSurfaceVariant
                                )

                                Text(
                                    text = item.timestamp,
                                    style = MaterialTheme.typography.labelSmall,
                                    color = MaterialTheme.colorScheme.onSurfaceVariant
                                )

                                Text(
                                    text = "View details →",
                                    style = MaterialTheme.typography.labelMedium,
                                    fontWeight = FontWeight.Bold,
                                    color = MaterialTheme.colorScheme.primary
                                )
                            }
                        }
                    }
                }
            }
        }

        Spacer(modifier = Modifier.height(8.dp))
    }

    // HISTORY DETAIL POPUP

    selectedHistoryItem?.let { item ->

        AlertDialog(
            onDismissRequest = {
                selectedHistoryItem = null
            },
            title = {
                Text(
                    text = when (item.module) {
                        "URL" -> "Phishing Scan Details"
                        "SMS" -> "SMS Scan Details"
                        "Network" -> "Network Scan Details"
                        else -> "Scan Details"
                    },
                    fontWeight = FontWeight.Bold
                )
            },
            text = {
                Column(
                    modifier = Modifier
                        .heightIn(max = 380.dp)
                        .verticalScroll(rememberScrollState()),
                    verticalArrangement = Arrangement.spacedBy(10.dp)
                ) {
                    Text(
                        text = "ORIGINAL INPUT",
                        style = MaterialTheme.typography.labelLarge,
                        fontWeight = FontWeight.Bold,
                        color = MaterialTheme.colorScheme.primary
                    )

                    Text(
                        text = item.input.ifBlank {
                            "Input details were not saved for this older scan."
                        },
                        style = MaterialTheme.typography.bodyMedium
                    )

                    Text(
                        text = "PREDICTION",
                        style = MaterialTheme.typography.labelLarge,
                        fontWeight = FontWeight.Bold,
                        color = MaterialTheme.colorScheme.primary
                    )

                    Text(
                        text = item.result,
                        style = MaterialTheme.typography.bodyMedium,
                        fontWeight = FontWeight.Bold
                    )

                    Text(
                        text = "SCAN SUMMARY",
                        style = MaterialTheme.typography.labelLarge,
                        fontWeight = FontWeight.Bold,
                        color = MaterialTheme.colorScheme.primary
                    )

                    Text("Items scanned: ${item.scannedCount}")
                    Text("Threats flagged: ${item.threatCount}")

                    Text(
                        text = "SCAN TIME",
                        style = MaterialTheme.typography.labelLarge,
                        fontWeight = FontWeight.Bold,
                        color = MaterialTheme.colorScheme.primary
                    )

                    Text(item.timestamp)
                }
            },
            confirmButton = {
                TextButton(
                    onClick = {
                        selectedHistoryItem = null
                    }
                ) {
                    Text("Close")
                }
            }
        )
    }
}

// COLOURFUL MODULE CARD

@Composable
private fun DashboardModuleCard(
    title: String,
    description: String,
    symbol: String,
    containerColor: androidx.compose.ui.graphics.Color,
    modifier: Modifier = Modifier,
    onClick: () -> Unit
) {
    Card(
        modifier = modifier
            .heightIn(min = 155.dp)
            .clickable(onClick = onClick),
        colors = CardDefaults.cardColors(
            containerColor = containerColor
        )
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(14.dp),
            verticalArrangement = Arrangement.spacedBy(9.dp)
        ) {
            Text(
                text = symbol,
                style = MaterialTheme.typography.headlineMedium
            )

            Text(
                text = title,
                style = MaterialTheme.typography.titleSmall,
                fontWeight = FontWeight.Bold,
                color = MaterialTheme.colorScheme.onSurface
            )

            Text(
                text = description,
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )

            Text(
                text = "Open module →",
                style = MaterialTheme.typography.labelMedium,
                fontWeight = FontWeight.Bold,
                color = MaterialTheme.colorScheme.primary
            )
        }
    }
}

// COLOURFUL STATISTICS CARD

@Composable
private fun StatisticCard(
    title: String,
    value: String,
    containerColor: androidx.compose.ui.graphics.Color,
    modifier: Modifier = Modifier
) {
    Card(
        modifier = modifier,
        colors = CardDefaults.cardColors(
            containerColor = containerColor
        )
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            Text(
                text = title,
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )

            Text(
                text = value,
                style = MaterialTheme.typography.headlineMedium,
                fontWeight = FontWeight.Bold,
                color = MaterialTheme.colorScheme.onSurface
            )
        }
    }
}
