import { Streamlit, RenderData } from "streamlit-component-lib"
import Two from "two.js"

// Setup two.js canvas in the container
const container = document.getElementById("container") as HTMLElement
const two = new Two({ fullscreen: true }).appendTo(container)

// Dummy rectangle for example
const rect = two.makeRectangle(two.width * 0.5, two.height * 0.5, 100, 100)
rect.fill = "#FF8000"

// Streamlit will call this function when the data is changed
function onRender(event: Event): void {
  const data = (event as CustomEvent<RenderData>).detail.args.data

  // Perform update logic here...
  document.getElementById("greeting")!.innerText = data.greeting

  // Update two.js objects
  two.update()
}

// Update timestamp for animation
let timestamp = 0
setInterval(() => {
  timestamp += 1

  // Send timestamp back to Streamlit
  Streamlit.setComponentValue(timestamp)
}, 1000)

Streamlit.events.addEventListener(Streamlit.RENDER_EVENT, onRender)
Streamlit.setComponentReady()
