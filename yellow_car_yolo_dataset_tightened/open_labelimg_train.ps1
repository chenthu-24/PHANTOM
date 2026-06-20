$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$ImageDir = Join-Path $Root "images\train"
$ClassFile = Join-Path $Root "predefined_classes.txt"
$SaveDir = Join-Path $Root "labels\train"

labelImg $ImageDir $ClassFile $SaveDir
