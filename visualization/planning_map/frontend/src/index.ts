import { Streamlit, RenderData } from "streamlit-component-lib"
import Two from "two.js"

// Setup two.js canvas in the container
const container = document.getElementById("container") as HTMLElement
const two = new Two({ fullscreen: true }).appendTo(container)
two.renderer.domElement.style.background = 'rgb(0, 128, 150)';

var stage = new Two.Group();
var shape = new Two.Rectangle(0, 0, 1, 1);
var offsets = undefined;
const polyTypes = [17, 1];
var lineTypes = [0];
const rectTypes = [99];
const scale = 5;
const lineWidth = 1;
const xReverse = -1;
var agentPoly = {};
var egoPoly = {};
const btnOffset = 70;
var isPlaying = false;
var currentFrame = 0;
var totalFrame = undefined;
var loadedData = undefined;
var playBtn = two.load(`images/icons/play.svg`, () => {});
var pauseBtn = two.load('images/icons/pause.svg', () => {});
var agentGroup = new Two.Group();
var mapGroup = new Two.Group();
var goalGroup = new Two.Group();
var percCirclesGroup = new Two.Group();
var egoGroup = new Two.Group();

var offsetFromLastFrame = [0, 0, 0];
var maxFrames = 80;
var lanes_to_mark = undefined;
var goal_pts = undefined;
var currentScene = 0;
var egoId = 'ego';

const queryString = window.location.search;
const urlParams = new URLSearchParams(queryString);
const taskName = urlParams.get('task');
const simName = urlParams.get('sim');
const sceneId = urlParams.get('sceneid');
const fileId = urlParams.get('fileid');
const datasetName = 'NuPlan';
//const datasetNameWithMap = simName.split("_")[1];
//const datasetName = datasetNameWithMap.split("-")[0];
var loadDemo = true;
var colorsForMap = {
  4: 'white',           // CROSSWALK (white)
  2: '#FF5733',         // STOP_LINE (orange)
  14: '#808080',        // WALKWAYS (gray)
  15: '#5B8CFF',        // CARPARK_AREA (light blue)
  17: '#A9A9A9',        // ROADBLOCK (dark gray)
  1: '#A9A9A9',          // INTERSECTION (default color, dark gray)
  18: 'green',
};
var colorsForRoute = {4: 'purple', 2: 'green', 14: 'yellow', 15: 'cyan', 17:'#FFA726', 18:'#FFA726'};
var colors = 'black';
var ego_fill = false;

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
