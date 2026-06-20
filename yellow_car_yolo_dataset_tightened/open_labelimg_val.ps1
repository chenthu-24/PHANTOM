$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$ImageDir = Join-Path $Root "images\val"
$ClassFile = Join-Path $Root "predefined_classes.txt"
$SaveDir = Join-Path $Root "labels\val"

labelImg $ImageDir $ClassFile $SaveDir
