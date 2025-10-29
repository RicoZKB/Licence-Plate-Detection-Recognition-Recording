#!/bin/bash
# Test script for gate flip functionality

echo "================================"
echo "Gate Flip Fix - Testing Guide"
echo "================================"
echo ""
echo "This script helps you test the gate flip fix in both versions."
echo ""
echo "TESTING PROCEDURE:"
echo "1. Run one of the versions below"
echo "2. Wait for a vehicle to enter (IN event logged)"
echo "3. Press 'o' to flip gate orientation"
echo "4. Observe the behavior and terminal output"
echo "5. Watch for OUT event when vehicle crosses gate"
echo ""
echo "Choose version to test:"
echo ""
echo "1) per7_format.py   - Simple complete reset (all vehicles exit on flip)"
echo "2) per8_optimized.py - Intelligent recalculation (smart exit based on position)"
echo ""
read -p "Enter choice (1 or 2): " choice

case $choice in
    1)
        echo ""
        echo "Starting per7_format.py (Simple Reset Version)..."
        echo "Press 'o' during runtime to flip gate orientation"
        echo "Press ESC to quit"
        echo ""
        sleep 2
        python3 per7_format.py
        ;;
    2)
        echo ""
        echo "Starting per8_optimized.py (Intelligent Recalculation Version)..."
        echo "Press 'o' during runtime to flip gate orientation"
        echo "Press ESC to quit"
        echo ""
        sleep 2
        python3 per8_optimized.py
        ;;
    *)
        echo "Invalid choice. Exiting."
        exit 1
        ;;
esac
