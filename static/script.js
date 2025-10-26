const videoPlayer = document.getElementById("videoPlayer");
const yoloResult = document.getElementById("yoloResult");

// Dummy chart for anomaly graph
const ctx = document.getElementById("anomalyGraph").getContext("2d");
const graph = new Chart(ctx, {
    type: "line",
    data: {
        labels: [],
        datasets: [{
            label: "Anomaly Score",
            data: [],
            borderColor: "red",
            borderWidth: 2
        }]
    },
    options: {
        responsive: true,
        scales: {
            y: { min: 0, max: 1 }
        }
    }
});

async function playDemo(name) {
    const response = await fetch("/get_video", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({ name })
    });

    const data = await response.json();
    if (data.error) {
        alert(data.error);
        return;
    }

    // Load video
    videoPlayer.src = "file:///" + data.path;
    yoloResult.innerText = `Playing demo: ${name}`;
}
