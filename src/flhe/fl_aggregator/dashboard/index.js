// Add this near the top of the file with other global variables
let serverPort = 8080; // Default port
let metadataDatasite = ""; // Placeholder for the datasite
let logIntervals = {};
let currentSortColumn = "status";
let currentSortDirection = "desc";
let completedProjectLogs = new Set();



// Function to save the port to localStorage
function savePort() {
  try {
    const portInput = document.getElementById("server-port").value;
    localStorage.setItem("serverPort", portInput);
    serverPort = portInput;
    checkServerStatus();
    fetchMetadata();
  } catch (error) {
    console.error("Error accessing localStorage:", error);
  }
}

// Retrieve saved port from localStorage on page load
document.addEventListener("DOMContentLoaded", () => {
  try {
    const savedPort = localStorage.getItem("serverPort");
    if (savedPort) {
      serverPort = savedPort;
      document.getElementById("server-port").value = savedPort;
    }
  } catch (error) {
    console.error("Error accessing localStorage:", error);
  }

  fetchMetadata();
  checkServerStatus();

});

document.getElementById("server-port").addEventListener("input", (event) => {
  serverPort = event.target.value;
  checkServerStatus();
  fetchMetadata();
});

async function fetchMetadata() {
  try {
    const response = await fetch(`http://localhost:${serverPort}/metadata`);
    if (response.ok) {
      const data = await response.json();
      metadataDatasite = data.datasite || "";
      console.log("Metadata datasite:", metadataDatasite);
      document.getElementById("metadata-datasite").textContent =
        metadataDatasite;
    }
  } catch (error) {
    console.error("Error fetching metadata:", error);
    document.getElementById("metadata-datasite").textContent = "";
  }
}


async function checkServerStatus() {
  const serverStatus = document.getElementById("server-status");
  try {
    const response = await fetch(`http://localhost:${serverPort}/apps/`);
    if (response.ok) {
      serverStatus.textContent = "✔️";
      serverStatus.classList.remove("error");
      serverStatus.classList.add("success");
    } else {
      throw new Error("Non-200 response");
    }
  } catch {
    serverStatus.textContent = "❌";
    serverStatus.classList.remove("success");
    serverStatus.classList.add("error");
  }
}

