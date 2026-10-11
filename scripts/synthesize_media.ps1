# Local Windows speech synthesis; never contacts a hosted TTS service.
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Speech
$mediaRoot = Join-Path $PSScriptRoot '..\data\media'
$segments = Get-Content -LiteralPath (Join-Path $mediaRoot 'audio\speech_script.json') -Raw | ConvertFrom-Json
$speaker = New-Object System.Speech.Synthesis.SpeechSynthesizer
try {
    $speaker.SelectVoice('Microsoft David Desktop')
    $speaker.Rate = 0
    $format = [System.Speech.AudioFormat.SpeechAudioFormatInfo]::new(16000, [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, [System.Speech.AudioFormat.AudioChannel]::Mono)
    foreach ($segment in $segments) {
        $target = Join-Path $mediaRoot ('audio\speech_' + $segment.id + '.wav')
        $speaker.SetOutputToWaveFile($target, $format)
        $speaker.Speak($segment.text)
        $speaker.SetOutputToNull()
    }
} finally {
    $speaker.Dispose()
}
