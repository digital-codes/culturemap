<?php

require_once __DIR__ . '/kekabo.php';

// Load configuration
$configFile = file_exists("/var/www/files/culturemap/config.ini")
    ? "/var/www/files/culturemap/config.ini"
    : __DIR__ . '/config.ini';

if (file_exists($configFile)) {
    $config = parse_ini_file($configFile, true);
    $apiKey = $config['agent']['api_key'] ?? 'YOUR_MISTRAL_API_KEY';
    $agentId = $config['agent']['agent_id'] ?? 'YOUR_AGENT_ID';
} else {
    $apiKey = 'YOUR_MISTRAL_API_KEY';
    $agentId = 'YOUR_AGENT_ID';
}

$endpoint = 'https://api.mistral.ai/v1/conversations';

/**
 * Return envelope for all public entry points.
 * error: 0 = ok, non-zero = failure
 */
function resultEnvelope(int $error, string $text, int $sessionId = 0, ?string $conversationId = null, array $raw = []): array {
    return [
        'error' => $error,
        'text' => ($text !== '' ? $text : 'Ich habe leider keine Antwort erhalten.'),
        'session_id' => $sessionId,
        'conversation_id' => $conversationId,
        'raw' => $raw,
    ];
}

/**
 * HTTP handler style entry point.
 *
 * Expected request fields:
 * - query (string)
 * - language (string, optional)
 * - session_id (int, required)
 * - conversation_id (string, optional)
 */
function handleHttpChatRequest(array $request): array {
    $query = isset($request['query']) ? trim((string)$request['query']) : '';
    $language = isset($request['language']) ? (string)$request['language'] : 'Deutsch';
    $sessionId = isset($request['session_id']) ? (int)$request['session_id'] : 0;

    // Keep conversation_id separate from session_id.
    $conversationId = isset($request['conversation_id']) ? (string)$request['conversation_id'] : null;
    if ($conversationId === '') {
        $conversationId = null;
    }

    if ($query === '') {
        return resultEnvelope(10, 'Fehlende Eingabe: query ist leer.', $sessionId, $conversationId);
    }

    // A session_id of 0 starts a fresh chat; we may clear conversation_id.
    if ($sessionId === 0) {
        $conversationId = null;
    }

    $turn = runChatTurn($query, $language, $conversationId);
    if (($turn['error'] ?? 99) !== 0) {
        return resultEnvelope((int)$turn['error'], (string)($turn['text'] ?? ''), $sessionId, $turn['conversation_id'] ?? $conversationId, $turn['raw'] ?? []);
    }

    return resultEnvelope(0, (string)$turn['text'], $sessionId, $turn['conversation_id'] ?? $conversationId, $turn['raw'] ?? []);
}

/**
 * Core chat turn runner.
 * - Starts a conversation if $conversationId is null
 * - Continues a conversation otherwise
 * - Properly handles function.call outputs by sending function.result and collecting the final message.output
 */
function runChatTurn(string $query, string $language, ?string $conversationId = null): array {
    try {
        $responseData = ($conversationId === null)
            ? sendInitialRequest($query, $language)
            : sendFollowUpRequest($conversationId, $query);

        if (!is_array($responseData)) {
            return ['error' => 21, 'text' => 'Ungültige Antwort vom Agenten.', 'conversation_id' => $conversationId, 'raw' => []];
        }

        // If the API returned a conversation_id, keep it.
        $cid = $responseData['conversation_id'] ?? $conversationId;

        $final = processAgentResponseCollectingText($responseData);
        $text = $final['text'] ?? '';
        $cid = $final['conversation_id'] ?? $cid;

        return ['error' => 0, 'text' => $text, 'conversation_id' => $cid, 'raw' => $final['raw'] ?? $responseData];
    } catch (Throwable $e) {
        return ['error' => 99, 'text' => 'Interner Fehler: ' . $e->getMessage(), 'conversation_id' => $conversationId, 'raw' => []];
    }
}

/**
 * Send initial request to start a conversation
 */
function sendInitialRequest(string $query, string $language): array {
    global $endpoint, $apiKey, $agentId;
    
    $currentDate = date('d.m.Y');
    $fullQuery = "Heute ist der $currentDate. Die Sprache des Nutzers ist $language. $query";
    
    $payload = [
        'inputs' => [
            [
                'role' => 'user',
                'content' => $fullQuery,
                'object' => 'entry',
                'type' => 'message.input'
            ]
        ],
        'stream' => false,
        'agent_id' => $agentId
    ];
    
    $curl = sendCurlRequest($endpoint, $payload, $apiKey);
    if (($curl['error'] ?? 0) !== 0) {
        return ['error' => $curl['error'], 'message' => $curl['message'] ?? 'Curl Fehler', 'raw' => $curl];
    }
    $responseData = json_decode($curl['body'] ?? '', true);
    return is_array($responseData) ? $responseData : ['error' => 22, 'message' => 'Antwort ist kein JSON.', 'raw' => $curl];
}

/**
 * Send follow-up request to existing conversation
 */
function sendFollowUpRequest(string $conversationId, string $query): array {
    global $endpoint, $apiKey;
    
    $url = "$endpoint/$conversationId";
    
    $payload = [
        'inputs' => $query,
        'stream' => false,
        'store' => true,
        'handoff_execution' => 'server'
    ];
    
    $curl = sendCurlRequest($url, $payload, $apiKey);
    if (($curl['error'] ?? 0) !== 0) {
        return ['error' => $curl['error'], 'message' => $curl['message'] ?? 'Curl Fehler', 'raw' => $curl];
    }
    $responseData = json_decode($curl['body'] ?? '', true);
    return is_array($responseData) ? $responseData : ['error' => 22, 'message' => 'Antwort ist kein JSON.', 'raw' => $curl];
}

/**
 * Generic CURL request wrapper
 */
function sendCurlRequest(string $url, array $payload, string $apiKey): array {
    $ch = curl_init($url);
    curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
    curl_setopt($ch, CURLOPT_POST, true);
    curl_setopt($ch, CURLOPT_POSTFIELDS, json_encode($payload));
    curl_setopt($ch, CURLOPT_HTTPHEADER, [
        'Content-Type: application/json',
        "Authorization: Bearer $apiKey"
    ]);

    $body = curl_exec($ch);
    $httpCode = (int)curl_getinfo($ch, CURLINFO_HTTP_CODE);
    $curlErrNo = curl_errno($ch);
    $curlErr = $curlErrNo ? curl_error($ch) : '';
    curl_close($ch);

    if ($curlErrNo) {
        return ['error' => 1, 'message' => 'Curl error: ' . $curlErr, 'http_code' => $httpCode, 'body' => (string)$body];
    }
    if ($httpCode >= 400) {
        return ['error' => 2, 'message' => 'HTTP error: ' . $httpCode, 'http_code' => $httpCode, 'body' => (string)$body];
    }

    return ['error' => 0, 'http_code' => $httpCode, 'body' => (string)$body];
}

/**
 * Process agent response and handle function calls or message outputs.
 * Returns a final text (best effort) and the latest conversation_id.
 */
function processAgentResponseCollectingText(array $responseData): array {
    $text = '';
    $conversationId = $responseData['conversation_id'] ?? null;

    if (!isset($responseData['outputs']) || !is_array($responseData['outputs'])) {
        // Sometimes API errors are returned without outputs.
        $fallback = '';
        if (isset($responseData['message']) && is_string($responseData['message'])) {
            $fallback = $responseData['message'];
        } elseif (isset($responseData['error']) && is_string($responseData['error'])) {
            $fallback = $responseData['error'];
        }
        return ['text' => $fallback, 'conversation_id' => $conversationId, 'raw' => $responseData];
    }

    foreach ($responseData['outputs'] as $output) {
        if (!is_array($output) || !isset($output['type'])) {
            continue;
        }

        if ($output['type'] === 'function.call') {
            $followUp = handleFunctionCall($output, $conversationId);
            if (is_array($followUp)) {
                $conversationId = $followUp['conversation_id'] ?? $conversationId;
                $nested = processAgentResponseCollectingText($followUp);
                $conversationId = $nested['conversation_id'] ?? $conversationId;
                $nestedText = (string)($nested['text'] ?? '');
                if ($nestedText !== '') {
                    $text .= ($text !== '' ? "\n" : '') . $nestedText;
                }
                // Prefer returning the last raw response (after tool execution).
                $responseData = $nested['raw'] ?? $followUp;
            }
        } elseif ($output['type'] === 'message.output') {
            $msg = handleMessageOutput($output);
            if ($msg !== '') {
                $text .= ($text !== '' ? "\n" : '') . $msg;
            }
        }
    }

    return ['text' => $text, 'conversation_id' => $conversationId, 'raw' => $responseData];
}

/**
 * Handle function call from agent
 */
function handleFunctionCall(array $output, ?string $conversationId): ?array {
    // echo "Function call detected: {$output['name']}\n";
    // echo "Tool call ID: {$output['tool_call_id']}\n";

    $arguments = json_decode($output['arguments'], true);

    if ($output['name'] === 'get_upcoming_events') {
        return handleGetUpcomingEvents($arguments, $conversationId, $output['tool_call_id']);
    }

    return null;
}

/**
 * Handle get_upcoming_events function call
 */
function handleGetUpcomingEvents(array $arguments, ?string $conversationId, string $toolCallId): ?array {
    global $endpoint, $apiKey;
    /*
    echo "Fetching events for date range:\n";
    echo "Start date: {$arguments['start_date']}\n";
    echo "End date: {$arguments['end_date']}\n\n";
    */

    $timezone = new DateTimeZone('Europe/Berlin');
    $startDateTime = DateTime::createFromFormat('Y-m-d\TH:i:sP', $arguments['start_date'], $timezone) ?: DateTime::createFromFormat('Y-m-d\TH:i:s', $arguments['start_date'], $timezone);
    $endDateTime = DateTime::createFromFormat('Y-m-d\TH:i:sP', $arguments['end_date'], $timezone) ?: DateTime::createFromFormat('Y-m-d\TH:i:s', $arguments['end_date'], $timezone);

    if ($startDateTime === false || $endDateTime === false) {
        // echo "Error: Invalid date format in arguments\n";
        return null;
    }

    $startTimestamp = $startDateTime->getTimestamp();
    $endTimestamp = $endDateTime->getTimestamp();

    $events = getEvents($startTimestamp, $endTimestamp);
    // echo "Fetched " . count($events) . " events.\n\n";

    $payload = [
        'inputs' => [
            [
                'tool_call_id' => $toolCallId,
                'result' => json_encode($events),
                'object' => 'entry',
                'type' => 'function.result'
            ]
        ],
        'stream' => false,
        'store' => true,
        'handoff_execution' => 'server'
    ];

    if ($conversationId) {
        $url = "$endpoint/$conversationId";
        $curl = sendCurlRequest($url, $payload, $apiKey);
        if (($curl['error'] ?? 0) !== 0) {
            return ['error' => $curl['error'], 'message' => $curl['message'] ?? 'Curl Fehler', 'raw' => $curl];
        }
        $followUpData = json_decode($curl['body'] ?? '', true);
        /*
        echo "Follow-up Response:\n";
        print_r($followUpData);
        echo "\n";
        */
        return is_array($followUpData) ? $followUpData : null;
    }

    return null;
}

/**
 * Handle message output from agent
 */
function handleMessageOutput(array $output): string {
    $text = '';

    if (is_string($output['content'])) {
        $text = $output['content'];
    } elseif (is_array($output['content'])) {
        foreach ($output['content'] as $content) {
            if ($content['type'] === 'text') {
                $text .= $content['text'];
            } elseif ($content['type'] === 'tool_reference' && $content['tool'] === 'web_search') {
                //echo "Web Search Result: {$content['title']}\n";
                //echo "URL: {$content['url']}\n";
            }
        }
    }

    return $text;
}

/**
 * Test function that mimics an HTTP request handler:
 * - First call uses session_id = 0 to start a new chat
 * - Second call uses session_id != 0 and reuses conversation_id
 */
function testChat(): void {
    echo "=== TEST 1: HTTP-style Initial Request (session_id=0) ===\n";
    $r1 = handleHttpChatRequest([
        'session_id' => 0,
        'conversation_id' => '',
        'language' => 'Deutsch',
        'query' => 'Am Wochenende soll die Eröffnung der Le Cage Ausstellung sein. wann genau?',
    ]);
    print_r($r1);
    echo "\n";
    if ($r1['error'] === 0) {
        echo "\n$r1[text]\n\n";
    }   

    $conversationId = $r1['conversation_id'] ?? null;

    echo "=== TEST 2: HTTP-style Follow-up Request (session_id=1) ===\n";
    $r2 = handleHttpChatRequest([
        'session_id' => 1,
        'conversation_id' => $conversationId,
        'language' => 'Deutsch',
        'query' => 'Ist das eine Tanzveranstaltung?',
    ]);
    print_r($r2);

    if ($r2['error'] === 0) {
        echo "\n$r2[text]\n\n";
    }   

}

// Run tests only from CLI to avoid breaking HTTP handlers.
if (PHP_SAPI === 'cli') {
    testChat();
}
