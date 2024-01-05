import {Streamlit, RenderData} from "streamlit-component-lib";
import Two from "two.js";
import {ZUI} from "./zui.js";

// Setup two.js canvas in the container
var container = document.getElementById("container");
var two = new Two({fullscreen: true}).appendTo(container);
two.renderer.domElement.style.background = 'rgb(0, 128, 150)';

var shape = new Two.Rectangle(0, 0, 1, 1);
var offsets = [];
const polyTypes = [17, 1];
var lineTypes = [0, 11];
const rectTypes = [99];
const scale = 5;
const lineWidth = 1;
const xReverse = -1;
var agentPoly = {};
const btnOffset = 70;
var isPlaying = false;
var currentFrame = 0;
var totalFrame = 0;
var playBtn = two.load('./images/icons/play.svg');
var pauseBtn = two.load('./images/icons/pause.svg');
var nextFrameBtn = two.load('./images/icons/next-frame.svg');
var nextScenarioBtn = two.load('./images/icons/next-scenario.svg');
var previousFrameBtn = two.load('./images/icons/previous-frame.svg');
var previousScenarioBtn = two.load('./images/icons/previous-scenario.svg');
var panel = two.makeRoundedRectangle(two.width / 2 + 65, two.height - 123, 390, 70, 35);
var stage = new Two.Group();
var agentGroup = new Two.Group();
var mapGroup = new Two.Group();
var routeGroup = new Two.Group();
var goalGroup = new Two.Group();
var trajectoryGroup = new Two.Group();
var percCirclesGroup = new Two.Group();
var egoGroup = new Two.Group();
var btnGroup = new Two.Group();

var offsetFromLastFrame = [0, 0, 0];
var maxFrames = 1500000;
var lanes_to_mark = [];
var goal_pts = [];
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
var colorsForRoute = {4: 'purple', 2: 'green', 14: 'yellow', 15: 'cyan', 17: '#FFA726', 18: '#FFA726'};
var ego_fill = false;

var agentDic = {};
var route_ids = [];
var roadDic = {};
// Update timestamp for animation
// let timestamp = 0;
let data_initialized = false;
const frequencyChange = 2;
var currentScenarioId;
var zui;


// Debounce function handling resize event for button positions
function debounce(func, wait) {
    var timeout;
    return function() {
        var context = this, args = arguments;
        var later = function() {
            timeout = null;
            func.apply(context, args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
};

// Adjusted resize event
var handleResize = debounce(function() {
    updateBtn();
    update(0);
    console.log('resizing');
}, 100); // Wait for 100 ms after resize event stops firing

// Bind the debounced resize event
two.bind('resize', handleResize);


function drawPoly(dic, selected, colors, targetGroup, alpha = 1.0) {
    const roadType = dic['type'];
    if (selected.includes(roadType)) {
        const pointsArray = new Array();
        const points = dic['xyz'];
        for (let xyzArray of points) {
            if (offsets.length === 0) {
                offsets = [-xyzArray[0], -xyzArray[1]];
            }
            pointsArray.push(new Two.Anchor((xyzArray[0] + offsets[0]) * scale * xReverse, (xyzArray[1] + offsets[1]) * scale));
        }
        var poly = two.makePath(pointsArray);
        // draw polygons
        // var poly = new two.makePath);
        poly.linewidth = lineWidth / 2;
        poly.opacity = 0.2;
        // Default color is black
        // Check if there is a color for this roadType
        if (colors.hasOwnProperty(roadType)) {
            const color = colors[roadType];
            poly.opacity = alpha;
            poly.fill = color;
        }
        targetGroup.add(poly);
    }
}

function drawALine(points, arrow, color, dash, targetGroup, continuous = false) {
    var ptCounter = 0;
    var prev_x = 0;
    var prev_y = 0;
    var interval = 0;
    if (dash) {
        interval = 10;
    }
    for (let xyzArray of points) {
        if (ptCounter === interval) {
            ptCounter = 0;
        } else {
            ptCounter++;
            continue;
        }
        if (prev_x === 0) {
            prev_x = (xyzArray[0] + offsets[0]) * scale * xReverse;
            prev_y = (xyzArray[1] + offsets[1]) * scale;
        } else {
            let x = (xyzArray[0] + offsets[0]) * scale * xReverse;
            let y = (xyzArray[1] + offsets[1]) * scale;
            if (arrow) {
                let line = two.makeArrow(prev_x, prev_y, x, y, 5);
                line.linewidth = lineWidth * 3;
                line.stroke = color;
                targetGroup.add(line);
            } else {
                let line = two.makeLine(prev_x, prev_y, x, y);
                line.linewidth = lineWidth;
                line.stroke = color;
                targetGroup.add(line);
            }
            prev_x = x;
            prev_y = y;
            if (!continuous) {
                prev_x = 0;
                prev_y = 0;
            }
        }
    }
}

function drawLanes(roads, targetGroup) {
    if (datasetName === 'NuPlan') {
        for (var key in roads) {
            const dic = roads[key];
            var roadType = dic['type'];
            if (Array.isArray(roadType)) {
                roadType = roadType[0];
            }
            const pointsToGo = dic['xyz'];
            if (lineTypes.includes(roadType)) {
                var to_mark = false;
                var colorToGo = 'white';
                var dashToGO = true;
                if (lanes_to_mark.includes(key)) {
                    to_mark = true;
                    colorToGo = 'black';
                    dashToGO = true;
                }
                drawALine(pointsToGo, to_mark, colorToGo, dashToGO, targetGroup);
            }
        }
    }
}

function drawMap(roads) {
    var colorsForMap = {
        4: 'white',           // CROSSWALK (white)
        2: '#FF5733',         // STOP_LINE (orange)
        14: '#808080',        // WALKWAYS (gray)
        15: '#5B8CFF',        // CARPARK_AREA (light blue)
        17: '#A9A9A9',        // ROADBLOCK (dark gray)
        1: '#A9A9A9',          // INTERSECTION (default color, dark gray)
        18: 'green',
    };
    for (var key in roads) {
        drawPoly(roads[key], polyTypes, colorsForMap, mapGroup);  // 17=1
    }
    //console.log('here')
    // draw lanes
    drawLanes(roads, mapGroup);
    // draw others
    for (var key in roads) {
        drawPoly(roads[key], [2, 4, 14, 15, 7, 8], colorsForMap, mapGroup);  // no 3, 5, 6
    }
    // draw parking lots
    for (var key in roads) {
        //console.log('thre')
        const dic = roads[key];
        const roadType = dic['type'];
        //console.log('map road',roadType)
        if (rectTypes.includes(roadType)) {
            const points = dic['xyz'];
            var rectShape = dic['shape'];
            const x = (points[0] + offsets[0]) * scale * xReverse;
            const y = (points[1] + offsets[1]) * scale;
            var rect = two.makeRectangle(x, y, rectShape[0] * scale, rectShape[1] * scale);
            rect.rotation = Math.PI / 2 - dic['dir'];
            rect.fill = 'white';
            rect.opacity = 0.3;
            mapGroup.add(rect);
        }
    }
    // draw circle
    var circle = two.makeCircle(two.width / 4, two.height / 4, 300 * scale);
    // circle.fill = 'gray';
    circle.opacity = 0.2;
    circle.stroke = 'red';
    percCirclesGroup.add(circle);
    circle = two.makeCircle(two.width / 4, two.height / 4, 150 * scale);
    // circle.fill = 'gray';
    circle.stroke = 'green';
    circle.opacity = 0.2;
    percCirclesGroup.add(circle);
    stage.add(percCirclesGroup);
    stage.add(mapGroup);
    two.add(stage);
}

function drawRoute(route_ls, road_data) {
    for (var route_id of route_ls) {
        drawPoly(road_data[route_id], [17, 18], colorsForRoute, routeGroup, 0.4);  // 17=1
    }
    stage.add(routeGroup);
    two.add(stage);
}

function drawEgoPose(ego_dic, ego_fill = 'train') {
    var prev_x = 0;
    var ego_states = ego_dic['pose'];
    var ego_intentions = ego_dic['intention'];
    for (let i = 0; i < ego_states.length; i += 5) {
        let [x, y, z, yaw] = ego_states[i];
        if (x == -1) {continue;}
        if (Math.abs(x - prev_x) < 0.1) {continue;}
        let [w, h] = [2.297, 5.176];
        var pointsArray = new Array();
        pointsArray.push(new Two.Anchor((x + offsets[0]) * scale * xReverse - w * scale / 2, (y + offsets[1]) * scale - h * scale / 2));
        pointsArray.push(new Two.Anchor((x + offsets[0]) * scale * xReverse, (y + offsets[1]) * scale - h * scale / 2 * 1.3));
        pointsArray.push(new Two.Anchor((x + offsets[0]) * scale * xReverse + w * scale / 2, (y + offsets[1]) * scale - h * scale / 2));
        pointsArray.push(new Two.Anchor((x + offsets[0]) * scale * xReverse + w * scale / 2, (y + offsets[1]) * scale + h * scale / 2));
        pointsArray.push(new Two.Anchor((x + offsets[0]) * scale * xReverse - w * scale / 2, (y + offsets[1]) * scale + h * scale / 2));
        var poly = two.makePath(pointsArray);

        var color = 'gray';
        // 0: turn left, 1: turn right, 2: accelerate, 3: decelerate, 4: idle
        if (ego_intentions !== undefined) {
            if (ego_intentions[i] == 0) {
                color = 'blue';
            } else if (ego_intentions[i] == 1) {
                color = 'yellow';
            } else if (ego_intentions[i] == 2) {
                color = 'pink';
            } else if (ego_intentions[i] == 3) {
                color = 'red';
            } else if (ego_intentions[i] == 4) {
                color = 'gray';
            }
        }

        if (ego_fill == 'train') {
            poly.fill = color;
            poly.opacity = 0.3;
        } else if (ego_fill == 'test') {
            // poly.fill = 'yellow';
            poly.opacity = 0.3;
        }
        //console.log('here')
        poly.rotation = -yaw - Math.PI / 2;
        egoGroup.add(poly);
    }
    stage.add(egoGroup);
    two.add(stage);
}


function setOffset(agentDic, route_ids) {
    // if (datasetName == 'Waymo'){
    //   egoId = loadedData['predicting']['ego_id'][1];
    //   var ego = loadedData['agent'][egoId];
    // }
    if (agentDic !== {}) {
        egoId = 'ego';
        let ego = agentDic[egoId];
        let [x, y, z, yaw] = ego['pose'][currentFrame];
        if (offsets !== undefined) {
            let dx = -x - two.width / 4 / scale - offsets[0];
            let dy = -y + two.height / 4 / scale - offsets[1];
            // var dyaw = -yaw - offsets[2];
            // offsetFromLastFrame = [dx, dy, dyaw];
            offsetFromLastFrame = [dx, dy, 0];
        }
        offsets = [-x - two.width / 4 / scale, -y + two.height / 4 / scale, 0];
        if (lanes_to_mark.length === 0) {
            if (route_ids !== undefined) {
                for (const lane_id of route_ids) {
                    if (!lanes_to_mark.includes(lane_id)) {
                        lanes_to_mark.push(lane_id);
                    }
                }
                console.log('drawing routes: ', route_ids);
            }
        }
    }
}


var FRAMERATE = 10;  //10fps

function update(frameCount) {
    if (isPlaying) {
        playBtn.opacity = 0;
        pauseBtn.opacity = 1;
    } else {
        playBtn.opacity = 1;
        pauseBtn.opacity = 0;
    }
    if (data_initialized) {
        if (isPlaying && totalFrame !== 0 && frameCount % FRAMERATE == 0) {
            console.log('updating: ', currentFrame, totalFrame);
            if (currentFrame < totalFrame && currentFrame < maxFrames) {
                setOffset(agentDic, route_ids);
                for (var key in agentPoly) {
                    if (agentDic[key] === undefined) {
                        continue;
                    }
                    if (agentDic[key]['ending_frame'] === -1 || (currentFrame >= agentDic[key]['starting_frame'] / frequencyChange && currentFrame < agentDic[key]['ending_frame'] / frequencyChange)) {
                        const frame_index = currentFrame - Math.ceil(agentDic[key]['starting_frame'] / frequencyChange);
                        let poly = agentPoly[key];
                        poly.opacity = 1;
                        if (frame_index >= agentDic[key]['pose'].length) {
                            console.log('out index: ', frame_index, currentFrame, agentDic[key]['starting_frame'] / frequencyChange, agentDic[key]['pose'].length, agentDic[key]['ending_frame'] / frequencyChange);
                            continue;
                        }
                        // console.log('inspect: ', frame_index, currentFrame, agentDic[key]['starting_frame']/frequencyChange, agentDic[key]['pose'].length, agentDic[key]['ending_frame']/frequencyChange);
                        let [x, y, z, yaw] = agentDic[key]['pose'][frame_index];
                        // [w, h] = agents[key]['shape'][currentFrame];
                        let [w, h] = agentDic[key]['shape'][frame_index];
                        poly.position.x = (x + offsets[0]) * scale * xReverse;
                        poly.position.y = (y + offsets[1]) * scale;
                        poly.rotation = -yaw - Math.PI / 2;
                    } else {
                        let poly = agentPoly[key];
                        poly.opacity = 0;
                    }
                }
                mapGroup.position.x += offsetFromLastFrame[0] * scale * xReverse;
                mapGroup.position.y += offsetFromLastFrame[1] * scale;
                routeGroup.position.x += offsetFromLastFrame[0] * scale * xReverse;
                routeGroup.position.y += offsetFromLastFrame[1] * scale;
                goalGroup.position.x += offsetFromLastFrame[0] * scale * xReverse;
                goalGroup.position.y += offsetFromLastFrame[1] * scale;
                egoGroup.position.x += offsetFromLastFrame[0] * scale * xReverse;
                egoGroup.position.y += offsetFromLastFrame[1] * scale;
                trajectoryGroup.position.x += offsetFromLastFrame[0] * scale * xReverse;
                trajectoryGroup.position.y += offsetFromLastFrame[1] * scale;
                currentFrame++;
            } else {
                console.log('stop playing: ', currentFrame, totalFrame, maxFrames);
                isPlaying = false;
            }
        }
    }
    // console.log(frameCount);
    // if (frameCount > 50){
    //   two.pause();
    // }
}

function addZUI() {
    var domElement = two.renderer.domElement;
    // if (zui !== undefined) {
    //     zui.reset();
    // }
    // else{zui = new ZUI(stage);}
    zui = new ZUI(stage);
    var mouse = new Two.Vector();
    var touches = {};
    var distance = 0;
    var dragging = false;

    zui.zoomBy(0.7, 0, 0);
    zui.addLimits(0.06, 8);

    domElement.addEventListener('mousedown', mousedown, false);
    domElement.addEventListener('mousewheel', mousewheel, false);
    domElement.addEventListener('wheel', mousewheel, false);

    domElement.addEventListener('touchstart', touchstart, false);
    domElement.addEventListener('touchmove', touchmove, false);
    domElement.addEventListener('touchend', touchend, false);
    domElement.addEventListener('touchcancel', touchend, false);

    function mousedown(e) {
        mouse.x = e.clientX;
        mouse.y = e.clientY;
        // process btn actions
        if (mouse.x < (two.width / 2 + 50 + btnOffset / 2 * 1.2) && mouse.x > (two.width / 2 + 50 - btnOffset / 2 * 0.2)
            && mouse.y > (two.height - 140 - 30) && mouse.y < (two.height - 140 + 30)) {
            isPlaying = !isPlaying;
            if (!isPlaying) {Streamlit.setComponentValue(currentFrame);}
        } else if (mouse.x < (two.width / 2 + 50 + btnOffset / 2 * 1.2 + btnOffset) && mouse.x > (two.width / 2 + 50 - btnOffset / 2 * 0.2 + btnOffset)
            && mouse.y > (two.height - 140 - 30) && mouse.y < (two.height - 140 + 30)) {
            currentFrame = Math.min(currentFrame + 80, totalFrame - 1);
            isPlaying = true;
            update(0);
            isPlaying = false;
        } else if (mouse.x < (two.width / 2 + 50 + btnOffset / 2 * 1.2 + btnOffset * 2) && mouse.x > (two.width / 2 + 50 - btnOffset / 2 * 0.2 + btnOffset * 2)
            && mouse.y > (two.height - 140 - 30) && mouse.y < (two.height - 140 + 30)) {
            currentFrame = totalFrame - 2;
            isPlaying = true;
            update(0);
            isPlaying = false;
        } else if (mouse.x < (two.width / 2 + 50 + btnOffset / 2 * 1.2 - btnOffset) && mouse.x > (two.width / 2 + 50 - btnOffset / 2 * 0.2 - btnOffset)
            && mouse.y > (two.height - 140 - 30) && mouse.y < (two.height - 140 + 30)) {
            currentFrame = Math.max(currentFrame - 80, 0);
            isPlaying = true;
            update(0);
            isPlaying = false;
        } else if (mouse.x < (two.width / 2 + 50 + btnOffset / 2 * 1.2 - btnOffset * 2) && mouse.x > (two.width / 2 + 50 - btnOffset / 2 * 0.2 - btnOffset * 2)
            && mouse.y > (two.height - 140 - 30) && mouse.y < (two.height - 140 + 30)) {
            currentFrame = 0;
            isPlaying = true;
            update(0);
            isPlaying = false;
        } else {
            var rect = shape.getBoundingClientRect();
            dragging = mouse.x > rect.left && mouse.x < rect.right
                && mouse.y > rect.top && mouse.y < rect.bottom;
            window.addEventListener('mousemove', mousemove, false);
            window.addEventListener('mouseup', mouseup, false);
        }
    }

    function mousemove(e) {
        var dx = e.clientX - mouse.x;
        var dy = e.clientY - mouse.y;
        if (dragging) {
            shape.position.x += dx / zui.scale;
            shape.position.y += dy / zui.scale;
        } else {
            zui.translateSurface(dx, dy);
        }
        mouse.set(e.clientX, e.clientY);
    }

    function mouseup(e) {
        window.removeEventListener('mousemove', mousemove, false);
        window.removeEventListener('mouseup', mouseup, false);
    }

    function mousewheel(e) {
        var dy = (e.wheelDeltaY || -e.deltaY) / 1000;
        zui.zoomBy(dy, e.clientX, e.clientY);
    }

    function touchstart(e) {
        switch (e.touches.length) {
            case 2:
                pinchstart(e);
                break;
            case 1:
                panstart(e)
                break;
        }
    }

    function touchmove(e) {
        switch (e.touches.length) {
            case 2:
                pinchmove(e);
                break;
            case 1:
                panmove(e)
                break;
        }
    }

    function touchend(e) {
        touches = {};
        var touch = e.touches[0];
        if (touch) {  // Pass through for panning after pinching
            mouse.x = touch.clientX;
            mouse.y = touch.clientY;
        }
    }

    function panstart(e) {
        var touch = e.touches[0];
        mouse.x = touch.clientX;
        mouse.y = touch.clientY;
    }

    function panmove(e) {
        var touch = e.touches[0];
        var dx = touch.clientX - mouse.x;
        var dy = touch.clientY - mouse.y;
        zui.translateSurface(dx, dy);
        mouse.set(touch.clientX, touch.clientY);
    }

    function pinchstart(e) {
        for (var i = 0; i < e.touches.length; i++) {
            var touch = e.touches[i];
            touches[touch.identifier] = touch;
        }
        var a = touches[0];
        var b = touches[1];
        var dx = b.clientX - a.clientX;
        var dy = b.clientY - a.clientY;
        distance = Math.sqrt(dx * dx + dy * dy);
        mouse.x = dx / 2 + a.clientX;
        mouse.y = dy / 2 + a.clientY;
    }

    function pinchmove(e) {
        for (var i = 0; i < e.touches.length; i++) {
            var touch = e.touches[i];
            touches[touch.identifier] = touch;
        }
        var a = touches[0];
        var b = touches[1];
        var dx = b.clientX - a.clientX;
        var dy = b.clientY - a.clientY;
        var d = Math.sqrt(dx * dx + dy * dy);
        var delta = d - distance;
        zui.zoomBy(delta / 250, mouse.x, mouse.y);
        distance = d;
    }
}

function updateBtn(){
    panel.position.set(two.width / 2 + 65, two.height - 123);
    playBtn.position.set(two.width / 2 + 50, two.height - 140);
    pauseBtn.position.set(two.width / 2 + 50, two.height - 140);
    nextFrameBtn.position.set(two.width / 2 + 50 + btnOffset, two.height - 140);
    nextScenarioBtn.position.set(two.width / 2 + 50 + btnOffset * 2, two.height - 140);
    previousFrameBtn.position.set(two.width / 2 + 50 - btnOffset, two.height - 140);
    previousScenarioBtn.position.set(two.width / 2 + 50 - btnOffset * 2, two.height - 140);
}

function addBtn() {
    panel.fill = 'white';
    btnGroup.add(panel);

    playBtn.position.set(two.width / 2 + 50, two.height - 140);
    playBtn.scale = 0.07;
    // playBtn.opacity = 1;
    btnGroup.add(playBtn);

    pauseBtn.position.set(two.width / 2 + 50, two.height - 140);
    pauseBtn.scale = 0.75;
    pauseBtn.opacity = 0;
    btnGroup.add(pauseBtn);

    nextFrameBtn.position.set(two.width / 2 + 50 + btnOffset, two.height - 140);
    nextFrameBtn.scale = 0.12;
    btnGroup.add(nextFrameBtn);

    nextScenarioBtn.position.set(two.width / 2 + 50 + btnOffset * 2, two.height - 140);
    nextScenarioBtn.scale = 0.07;
    btnGroup.add(nextScenarioBtn);

    previousFrameBtn.position.set(two.width / 2 + 50 - btnOffset, two.height - 140);
    previousFrameBtn.scale = 0.07;
    btnGroup.add(previousFrameBtn);

    previousScenarioBtn.position.set(two.width / 2 + 50 - btnOffset * 2, two.height - 140);
    previousScenarioBtn.scale = 0.07;
    btnGroup.add(previousScenarioBtn);

    // var previousScenarioBtn = two.load('./images/icons/previous-scenario.svg');
    // previousScenarioBtn.position.set(two.width / 2 + 50 - btnOffset * 2, two.height - 140);
    // previousScenarioBtn.scale = 0.07;
    // btnGroup.add(previousScenarioBtn);
    two.add(btnGroup);
}

function initAgents(agentDic, frameId) {
    totalFrame = agentDic[egoId]['pose'].length;
    for (var key in agentDic) {
        let [x, y, z, yaw] = agentDic[key]['pose'][frameId];
        // yaw = 0;
        if (x === -1) {
            continue;
        }
        let [w, h] = agentDic[key]['shape'][frameId];

        var pointsArray = new Array();
        pointsArray.push(new Two.Anchor((x + offsets[0]) * scale * xReverse - w * scale / 2, (y + offsets[1]) * scale - h * scale / 2));
        pointsArray.push(new Two.Anchor((x + offsets[0]) * scale * xReverse, (y + offsets[1]) * scale - h * scale / 2 * 1.3));
        pointsArray.push(new Two.Anchor((x + offsets[0]) * scale * xReverse + w * scale / 2, (y + offsets[1]) * scale - h * scale / 2));
        pointsArray.push(new Two.Anchor((x + offsets[0]) * scale * xReverse + w * scale / 2, (y + offsets[1]) * scale + h * scale / 2));
        pointsArray.push(new Two.Anchor((x + offsets[0]) * scale * xReverse - w * scale / 2, (y + offsets[1]) * scale + h * scale / 2));
        var poly = two.makePath(pointsArray);

        // poly = new Two.Rectangle((x+offsets[0])*scale*xReverse, (y+offsets[1])*scale, w*scale, h*scale);
        if (key == egoId) {
            poly.fill = 'white';
        } else {
            poly.fill = 'green';
        }
        // poly.rotation = -Math.PI/2+yaw;
        // Math.PI/2 - dic['dir'];
        // poly.rotation = yaw;
        poly.rotation = -yaw - Math.PI / 2;
        agentPoly[key] = poly;
        if (agentDic[key]['starting_frame'] !== 0) {
            poly.opacity = 0;
        }
        agentGroup.add(poly);
    }
    stage.add(agentGroup);
}

function drawTrajectory(poses, disc_size=0.35, interval=5) {
    if (poses !== undefined) {
        for (var i = 0; i < poses.length; i = i + interval) {
            let [x, y, z, yaw] = poses[i];
            // console.log('check prediction generation: ', x, y, offsets, scale);
            var circle = two.makeCircle((x + offsets[0]) * scale * xReverse, (y + offsets[1]) * scale, disc_size * scale);
            circle.fill = 'purple';
            circle.opacity = 0.5;
            trajectoryGroup.add(circle);
        }
        stage.add(trajectoryGroup);
        two.add(stage);
    }
}

// Streamlit will call this function when the data is changed
function onRender(event) {
    const data = event.detail.args.data;
    if (data.current_scenario_id !== undefined && data.agent_dic !== undefined && data.route_ids !== undefined && data.road_dic !== undefined) {
        if (currentScenarioId !== data.current_scenario_id) {
            // reset current scene
            const allGroups = [agentGroup, mapGroup, routeGroup, goalGroup, percCirclesGroup, egoGroup, trajectoryGroup, stage];
            for (let i = 0; i < allGroups.length; i++) {
                while (allGroups[i].children.length > 0) {
                    allGroups[i].remove(allGroups[i].children[0]);
                }
                allGroups[i].remove();
            }
            two.clear();
            stage = new Two.Group();
            agentGroup = new Two.Group();
            mapGroup = new Two.Group();
            routeGroup = new Two.Group();
            trajectoryGroup = new Two.Group();
            goalGroup = new Two.Group();
            percCirclesGroup = new Two.Group();
            egoGroup = new Two.Group();
            btnGroup = new Two.Group();
            container = document.getElementById("container");
            two = new Two({fullscreen: true}).appendTo(container);
            two.renderer.domElement.style.background = 'rgb(0, 128, 150)';

            currentScenarioId = data.current_scenario_id;
            console.log('reset and init with: ', currentScenarioId, data.route_ids);
            agentDic = JSON.parse(JSON.stringify(data.agent_dic));
            route_ids = JSON.parse(JSON.stringify(data.route_ids));
            roadDic = data.road_dic;
            // currentFrame = Math.floor(data.frame_index / 2) - 20;
            currentFrame = Math.floor(data.frame_index / 2);
            console.log('stepping to frame id: ', currentFrame);
            data_initialized = true;

            setOffset(agentDic, route_ids);
            drawMap(roadDic);
            drawRoute(route_ids, roadDic);
            if (agentDic['ego'] !== undefined) {
                drawEgoPose(agentDic['ego'])
            }
            initAgents(agentDic, 0);

            // TODO: update with frames and new predictions
            if (btnGroup.children.length === 0) {
                addBtn();
                two.bind('update', update);
                // isPlaying = true;
                // update(0);
                two.play();
                addZUI();
            } else {
                console.log('reset');
                // currentFrame = 0;
                two.play();
                addZUI();
                two.add(btnGroup);
            }
            isPlaying = true;
            update(0);
            isPlaying = false;
            Streamlit.setComponentValue(currentFrame);
        }
        if (data.prediction_generation !== undefined) {
            // draw trajectory
            console.log('draw trajectory: ', data.pred_kp_generation);
            stage.remove(trajectoryGroup);
            trajectoryGroup = new Two.Group();
            drawTrajectory(data.prediction_generation, 0.35);
            drawTrajectory(data.pred_kp_generation, 1, 1);
            two.add(btnGroup);
        }
    } else {
        console.log('data not ready ', data);
    }
}

// setInterval(() => {
//     if (data_initialized) {timestamp += 1}
//     // Send timestamp back to Streamlit
//     Streamlit.setComponentValue(timestamp)
// }, 1000)

Streamlit.events.addEventListener(Streamlit.RENDER_EVENT, onRender)
Streamlit.setComponentReady()
