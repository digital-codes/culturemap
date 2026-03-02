<?php


function getEvents($start, $end) {
    // Define the base URL of the Gancio instance
    $base_url = "https://keepkarlsruheboring.org";
    // Construct the API URL for fetching events
    $api_url = $base_url . "/api/events?start=" . $start . "&end=" . $end . "&show_recurrent=true";

    // Initialize cURL session
    $ch = curl_init();
    curl_setopt($ch, CURLOPT_URL, $api_url);
    curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
    curl_setopt($ch, CURLOPT_FOLLOWLOCATION, true);

    // Execute the cURL request
    $response = curl_exec($ch);

    // Check for errors
    if (curl_errno($ch)) {
        echo 'Curl error: ' . curl_error($ch);
        return [];
    }

    // Decode the JSON response
    $events = json_decode($response, true);

    // Check if the request was successful
    if ($events === null) {
        echo "Failed to fetch events.";
        return [];
    }

    $timezone = new DateTimeZone('Europe/Berlin');

    // Output the events
    foreach ($events as $event) {
        $start = new DateTime();
        $start->setTimestamp($event['start_datetime']);
        $start->setTimezone($timezone);
        $formatted_start = $start->format('d.m.Y H:i');

        $end = new DateTime();
        $end->setTimestamp($event['end_datetime']);
        $end->setTimezone($timezone);
        $formatted_end = $end->format('d.m.Y H:i');

        $eventList[] = array(
            "title" => $event['title'],
            "start" => $formatted_start ?? 'N/A',
            "end" => $formatted_end ?? 'N/A',
            "location" => $event['place']["name"] ?? 'N/A',
            "description" => $event['description'] ?? 'N/A'
        );
    }

    return $eventList;
}

// Calculate the date range for the next 7 days in Unix timestamps
$start_timestamp = time();
$end_timestamp = strtotime('+7 days');

$events = getEvents($start_timestamp, $end_timestamp);


print_r($events);

?>
