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
 * Main handler for HTTP requests
 */
function handleChatRequest(string $query, string $language,?string $conversationId = null): array {
    
    if ($conversationId === null) {
        return sendInitialRequest($query, $language);
    } else {
        return sendFollowUpRequest($conversationId, $query);
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
    
    $response = sendCurlRequest($endpoint, $payload, $apiKey);
    $responseData = json_decode($response, true);
    
    processAgentResponse($responseData);
    
    return $responseData;
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
    
    $response = sendCurlRequest($url, $payload, $apiKey);
    $responseData = json_decode($response, true);
    
    processAgentResponse($responseData);
    
    return $responseData;
}

/**
 * Generic CURL request wrapper
 */
function sendCurlRequest(string $url, array $payload, string $apiKey): string {
    $ch = curl_init($url);
    curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
    curl_setopt($ch, CURLOPT_POST, true);
    curl_setopt($ch, CURLOPT_POSTFIELDS, json_encode($payload));
    curl_setopt($ch, CURLOPT_HTTPHEADER, [
        'Content-Type: application/json',
        "Authorization: Bearer $apiKey"
    ]);

    $response = curl_exec($ch);

    if (curl_errno($ch)) {
        echo "Curl error: " . curl_error($ch) . "\n";
    }

    return $response;
}

/**
 * Process agent response and handle function calls or message outputs
 */
function processAgentResponse(array $responseData): void {
    if (!isset($responseData['outputs'])) {
        return;
    }

    $conversationId = $responseData['conversation_id'] ?? null;

    foreach ($responseData['outputs'] as $output) {
        if ($output['type'] === 'function.call') {
            handleFunctionCall($output, $conversationId);
        } elseif ($output['type'] === 'message.output') {
            handleMessageOutput($output);
        }
    }
}

/**
 * Handle function call from agent
 */
function handleFunctionCall(array $output, ?string $conversationId): void {
    // echo "Function call detected: {$output['name']}\n";
    // echo "Tool call ID: {$output['tool_call_id']}\n";

    $arguments = json_decode($output['arguments'], true);

    if ($output['name'] === 'get_upcoming_events') {
        handleGetUpcomingEvents($arguments, $conversationId, $output['tool_call_id']);
    }
}

/**
 * Handle get_upcoming_events function call
 */
function handleGetUpcomingEvents(array $arguments, ?string $conversationId, string $toolCallId): void {
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
        return;
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
        $response = sendCurlRequest($url, $payload, $apiKey);
        $followUpData = json_decode($response, true);
        /*
        echo "Follow-up Response:\n";
        print_r($followUpData);
        echo "\n";
        */
        processAgentResponse($followUpData);
    }
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
 * Test function for chat functionality
 */
function testChat(): void {
    echo "=== TEST 1: Initial Request ===\n";
    $response1 = handleChatRequest(
        "Am Wochenende soll die Eröffnung der Le Cage Ausstellung sein. wann genau?",
        "Deutsch",
        null,
    );
    print_r($response1);
    echo "\n";

    $conversationId = $response1['conversation_id'] ?? null;

    if ($conversationId) {
        echo "=== TEST 2: Follow-up Request ===\n";
        $response2 = handleChatRequest(
            "Ist das eine Tanzveranstaltung?",
            "Deutsch",
            $conversationId
        );
        print_r($response2);
    }
}

// Uncomment to run tests
testChat();
