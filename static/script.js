async function fetchData() {
    try {
        const response = await fetch('/api/sensor_data');
        if (response.ok) {
            const data = await response.json();

            document.getElementById("sensor-output").textContent = JSON.stringify(data.sensor_data, null, 2);
            document.getElementById("prediction-output").textContent = `Prediction: ${data.prediction}`;
            document.getElementById("confidence-output").textContent = `Confidence: ${(data.confidence * 100).toFixed(2)}%`;
        } else {
            document.getElementById("sensor-output").textContent = "No data available";
            document.getElementById("prediction-output").textContent = "Prediction: N/A";
            document.getElementById("confidence-output").textContent = "Confidence: N/A";
        }
    } catch (error) {
        console.error("Error fetching data:", error);
    }
}

// Update every 5 seconds
setInterval(fetchData, 5000);
fetchData();
