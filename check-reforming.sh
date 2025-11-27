#!/bin/bash


system_logs=$(grep -rilE '\breform(s|ed|ing)?\b' logs)

echo "those files have reforming action:"
for file in $system_logs; do
  echo -e "$file"
done

echo ""
echo ""
echo "Analyzing each file..."

for file in $system_logs; do
  echo ""
  echo "File: $file"
  grep -R "Reform" $file
done
