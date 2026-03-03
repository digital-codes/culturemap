<?php
declare(strict_types=1);

// Set the allowed origin and allowed methods

header("Access-Control-Allow-Headers: *");
header('Access-Control-Allow-Origin: *');
header('Access-Control-Allow-Methods: POST, OPTIONS');


// for validation
use Lcobucci\JWT\Token\Parser;
use Lcobucci\JWT\Validation\Validator;
use Lcobucci\JWT\Encoding\JoseEncoder;
use Lcobucci\JWT\Validation\Constraint\IdentifiedBy;
use Lcobucci\JWT\Validation\Constraint\LooseValidAt;
use Lcobucci\JWT\Encoding\CannotDecodeContent;
use Lcobucci\JWT\Token\InvalidTokenStructure;
use Lcobucci\JWT\Token\UnsupportedHeaderFound;
use Lcobucci\JWT\UnencryptedToken;
use Lcobucci\Clock\SystemClock;
use Lcobucci\JWT\Signer\Key\InMemory;
use Lcobucci\JWT\Signer\Rsa\Sha256;
use Lcobucci\JWT\Validation\Constraint\SignedWith;

//

require 'vendor/autoload.php';


function getConfig($local = false)
{
    $basedirFile = __DIR__ . '/basedir.txt';
    $basedir = '/var/www/files/';

    if (is_readable($basedirFile)) {
    $contents = @file_get_contents($basedirFile);
    if ($contents !== false) {
        $contents = trim($contents);
        // remove surrounding quotes if present
        $contents = trim($contents, "\"' \t\n\r\0\x0B");
        if ($contents !== '') {
        $basedir = $contents;
        }
    }
    }
    // ensure trailing slash
    $basedir = rtrim($basedir, "/\\") . '/';


    $public_file = "{$basedir}public.pem";
    $access_file = "{$basedir}access.ini";
    try {
        $publicKey = file_get_contents($public_file);
    } catch (Exception $e) {
        return [null, null];
    }
    $access = @parse_ini_file($access_file, true, INI_SCANNER_TYPED);
    if ($access === false) {
        return [null, null];
    }
    if (empty($access['JTI'])) {
        return [null, null];
    }
    // Define the jti claim for JWT
    $jti_claim = trim((string) $access['JTI']);

    return [$publicKey, $jti_claim];
}


function parseToken($token, $local = false)
{
    list($publicKey, $jti_claim) = getConfig($local);
    if (!checkToken($token, $publicKey, $jti_claim)) { 
        return null;
    }
    $parser = new Parser(new JoseEncoder());

    try {
        $tok = $parser->parse($token);
    } catch (CannotDecodeContent | InvalidTokenStructure | UnsupportedHeaderFound $e) {
        error_log($e);
        return null;
    }
    assert($tok instanceof UnencryptedToken);
    $user = $tok->claims()->get('uid');
    //echo("User: " . $user . PHP_EOL);
    return $user;
}

function checkToken($token, $publicKey, $jti_claim)
{
    $parser = new Parser(new JoseEncoder());
    $tok = $parser->parse($token);

    // Create a key object from the public key
    $check = InMemory::plainText($publicKey);

    // Verify the signature using the SHA-256 algorithm and the public key
    $validator = new Validator();
    $algorithm = new Sha256();

    if (!$validator->validate($tok, new SignedWith($algorithm, $check))) {
        error_log("Invalid token signature");
        return false;
    }

    $clock = new SystemClock(new DateTimeZone(date_default_timezone_get()));
    if (!$validator->validate($tok, new LooseValidAt($clock))) {
        error_log("Token is not valid");
        return false;
    }
    return true;
}



?>