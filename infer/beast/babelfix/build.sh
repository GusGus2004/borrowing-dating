#!/usr/bin/env bash
# Compile the babelfix BEAST2 package and install it into env/beast2-packages/babelfix.
# Needs a JDK 17 (javac). Uses $JAVA_HOME if set, else javac on PATH.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
PKG="$ROOT/env/beast2-packages"
JAVAC="${JAVA_HOME:+$JAVA_HOME/bin/}javac"
JAR="${JAVA_HOME:+$JAVA_HOME/bin/}jar"
CP="$PKG/archive/BEAST.base/2.7.8/lib/BEAST.base.jar:$PKG/archive/Babel/0.4.2/lib/Babel.addon.jar"

build="$(mktemp -d)"; trap 'rm -rf "$build"' EXIT
"$JAVAC" --release 17 -cp "$CP" -d "$build/classes" "$HERE"/src/babelfix/*.java
mkdir -p "$build/pkg/lib"
"$JAR" cf "$build/pkg/lib/babelfix.jar" -C "$build/classes" .
cp "$HERE/version.xml" "$build/pkg/"
rm -rf "$PKG/babelfix"
cp -R "$build/pkg" "$PKG/babelfix"
echo "installed babelfix -> $PKG/babelfix"
