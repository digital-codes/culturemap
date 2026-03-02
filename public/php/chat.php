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

// Initial agent request
$endpoint = 'https://api.mistral.ai/v1/conversations';


/**
 * Send initial request to start a conversation with the agent
 */
function sendInitialAgentRequest(string $endpoint, array $payload): string {
    global $agentId, $apiKey;
    $payload['stream'] = false;
    $payload['agent_id'] = $agentId;
    return sendCurlRequest($endpoint, $payload, $apiKey);
}

/**
 * Send follow-up request to the agent with tool results or new message
 */
function sendFollowUpAgentRequest(string $conversationId, array $payload): string {
    global $apiKey;
    $endpoint = "https://api.mistral.ai/v1/conversations/$conversationId";
    $payload['stream'] = false;
    $payload['store'] = true;
    $payload['handoff_execution'] = 'server';
    return sendCurlRequest($endpoint, $payload, $apiKey);
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
        'Authorization: Bearer ' . $apiKey
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
    global $apiKey;
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
 * Handle function call from agent (e.g., get_upcoming_events)
 */
function handleFunctionCall(array $output, string $conversationId): void {
    global $apiKey;
    echo "Function call detected: {$output['name']}\n";
    echo "Tool call ID: {$output['tool_call_id']}\n";

    $arguments = json_decode($output['arguments'], true);

    if ($output['name'] === 'get_upcoming_events') {
        handleGetUpcomingEvents($arguments, $conversationId, $output['tool_call_id']);
    }
}

/**
 * Handle get_upcoming_events function call
 */
function handleGetUpcomingEvents(array $arguments, string $conversationId, string $toolCallId): void {
    global $apiKey;
    echo "Fetching events for date range:\n";
    echo "Start date: {$arguments['start_date']}\n";
    echo "End date: {$arguments['end_date']}\n\n";

    $timezone = new DateTimeZone('Europe/Berlin');
    $startDateTime = DateTime::createFromFormat('Y-m-d\TH:i:sP', $arguments['start_date'], $timezone) ?: DateTime::createFromFormat('Y-m-d\TH:i:s', $arguments['start_date'], $timezone);
    $endDateTime = DateTime::createFromFormat('Y-m-d\TH:i:sP', $arguments['end_date'], $timezone) ?: DateTime::createFromFormat('Y-m-d\TH:i:s', $arguments['end_date'], $timezone);

    if ($startDateTime === false || $endDateTime === false) {
        echo "Error: Invalid date format in arguments\n";
        return;
    }

    $startTimestamp = $startDateTime->getTimestamp();
    $endTimestamp = $endDateTime->getTimestamp();

    $events = getEvents($startTimestamp, $endTimestamp);
    echo "Fetched " . count($events) . " events.\n\n";

    // Send results back to agent
    $payload = [
        'inputs' => [
            [
                'tool_call_id' => $toolCallId,
                'result' => json_encode($events),
                'object' => 'entry',
                'type' => 'function.result'
            ]
        ]
    ];

    $response = sendFollowUpAgentRequest($conversationId, $payload);
    $followUpData = json_decode($response, true);

    echo "Follow-up Response:\n";
    print_r($followUpData);
    echo "\n";

    // Process the follow-up response recursively
    processAgentResponse($followUpData);
}

/**
 * Handle message output from agent
 */
function handleMessageOutput(array $output): void {
    $text = '';

    if (is_string($output['content'])) {
        $text = $output['content'];
    } elseif (is_array($output['content'])) {
        foreach ($output['content'] as $content) {
            if ($content['type'] === 'text') {
                $text .= $content['text'];
            } elseif ($content['type'] === 'tool_reference' && $content['tool'] === 'web_search') {
                echo "Web Search Result: {$content['title']}\n";
                echo "URL: {$content['url']}\n";
            }
        }
    }

    if ($text) {
        echo "Agent Response:\n";
        echo "$text\n\n";
    }
}


// -----------------------------
$currentDate = date('d.m.Y');
$lang = "Deutsch";
$query = "Heute ist der $currentDate. Die Sprache des Nutzers ist $lang. ";
$query .= "Am Wochenende soll die Eröffnung der Le Cage Ausstellung sein. wann genau?";

$payload = [
    'inputs' => [
        [
            'role' => 'user',
            'content' => $query,
            'object' => 'entry',
            'type' => 'message.input'
        ]
    ],
    'stream' => false,
    'agent_id' => $agentId
];

$response = sendInitialAgentRequest($endpoint, $payload);
$responseData = json_decode($response, true);

echo "Initial Response:\n";
print_r($responseData);
echo "\n--------------------\n";

// Process response
processAgentResponse($responseData);

// Follow-up query
$followUpPayload = [
    'inputs' => 'Ist das eine Tanzveranstaltung?'
];

$conversationId = $responseData['conversation_id'] ?? null;
if ($conversationId) {
    $followUpResponse = sendFollowUpAgentRequest($conversationId, $followUpPayload);
    $followUpData = json_decode($followUpResponse, true);
    
    echo "Follow-up Response:\n";
    print_r($followUpData);
    echo "\n";
    
    processAgentResponse($followUpData);
}
