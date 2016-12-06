var constants = {
  margin_left: 300,
  margin_right: 50,
  margin_bottom: 50,
  min_feature_width: 3,
  min_gap_width: 1,
}

var state = {};
var mouseX = 0;

function parseURLParameters() {
  var match,
  pl     = /\+/g,  // Regex for replacing addition symbol with a space
  search = /([^&=]+)=?([^&]*)/g,
  decode = function (s) { return decodeURIComponent(s.replace(pl, " ")); },
  query  = window.location.search.substring(1);

  var urlParams = {};
  while (match = search.exec(query))
    urlParams[decode(match[1])] = decode(match[2]);

  if ("collapseAll" in urlParams)
    state.collapseAll = (urlParams["collapseAll"].toLowerCase() === "true");

  if ("resolution" in urlParams)
    state.resolution = Math.max(1, parseFloat(urlParams["resolution"]));
  // adjust zoom
  var zstart = constants.start;
  if ("start" in urlParams)
    zstart = Math.max(constants.start, parseFloat(urlParams["start"]));

  var zend = constants.end;
  if ("end" in urlParams)
    zend = Math.min(constants.end, parseFloat(urlParams["end"]));

  if(zstart < zend) {
    // set zoom to get: (zend - start) * zoom * scale = $("#timeline").width()
    var newZoom = $("#timeline").width() / state.scale / (zend - zstart);
    adjustZoom(newZoom, false);

    // set scrollLeft to get:  zstart * zoom * state.scale = scrollLeft()
    $("#timeline").scrollLeft(convertToPos(state, zstart));
  }

  if ("search" in urlParams) {
    state.searchEnabled = true;
    searchRegex = new Array(sizeHistory);
    currentPos = 0;
    nextPos = 1;
    searchRegex[currentPos] = new RegExp(urlParams["search"]);
  }
}

function makeTimelineTransparent() {
  state.timelineSvg.select("g#timeline").style("opacity", "0.1");
  $("#timeline").css("overflow-x", "hidden");
  state.timelineSvg.select("g#lines").style("opacity", "0.1");
  state.timelineSvg.select("g.locator").style("opacity", "0.1");
}

function makeTimelineOpaque() {
  state.timelineSvg.select("g#timeline").style("opacity", "1.0");
  $("#timeline").css("overflow-x", "scroll");
  state.timelineSvg.select("g#lines").style("opacity", "1.0");
  state.timelineSvg.select("g.locator").style("opacity", "1.0");
}

function mouseMoveHandlerWhenDown() {
  var p = d3.mouse(this);
  var select_block = state.timelineSvg.select("rect.select-block");
  var select_text = state.timelineSvg.select("text.select-block");
  var newWidth = Math.abs(p[0] - mouseX);
  select_block.attr("width", newWidth);
  if (p[0] >= mouseX) {
    select_block.attr("x", mouseX);
    select_text.attr("x", mouseX + (p[0] - mouseX) / 2);
  } else {
    select_block.attr("x", p[0]);
    select_text.attr("x", p[0] + (mouseX - p[0]) / 2);
  }
  var time = convertToTime(state, newWidth);
  select_text.text(getTimeString(time, time));
}

function mouseMoveHandlerWhenUp() {
  var p = d3.mouse(this);
  var x = parseFloat(p[0]);
  var scrollLeft = $("#timeline").scrollLeft();
  var paneWidth = $("#timeline").width();
  var currentTime = convertToTime(state, x);

  state.timelineSvg.select("g.locator").remove();
  var locator = state.timelineSvg.append("g").attr("class", "locator");
  locator.insert("line")
    .attr({
      x1: p[0],
      y1: 0,
      x2: p[0],
      y2: p[1] - state.thickness / 2,
      class: "locator",
    });
  locator.append("line")
    .attr({
      x1: p[0],
      y1: p[1] + state.thickness / 2,
      x2: p[0],
      y2: state.height,
      class: "locator",
    });
  var locatorText = locator.append("text");
  var text = getTimeString(currentTime, convertToTime(state, paneWidth));
  locatorText.attr("class", "locator").text(text)
  if ((x - scrollLeft) < paneWidth - 100) {
    locatorText.attr({x: x + 2, y: $(window).scrollTop() + 10});
    locatorText.attr("anchor", "start");
  }
  else {
    locatorText.attr({x: x - 2 - text.length * 7, y: $(window).scrollTop() + 10});
    locatorText.attr("anchor", "end");
  }
}

function mouseDownHandler() {
  state.timelineSvg.select("g.locator").remove();
  var p = d3.mouse(this);
  state.timelineSvg.append("rect")
    .attr({
      x : p[0],
      y : 0,
      class : "select-block",
      width : 0,
      height : state.height
    });
  state.timelineSvg.append("text")
    .attr({
      x : p[0],
      y : p[1],
      class : "select-block",
      anchor : "middle",
      "text-anchor" : "middle",
    }).text("0 us");
  mouseX = p[0];
  state.timelineSvg.on("mousemove", null);
  state.timelineSvg.on("mousemove", mouseMoveHandlerWhenDown);
  $(document).off("keydown");
}

function mouseUpHandler() {
  var p = d3.mouse(this);
  var select_block = state.timelineSvg.select("rect.select-block");
  var select_text = state.timelineSvg.select("text.select-block");
  var prevZoom = state.zoom;
  var selectWidth = parseInt(select_block.attr("width"));
  var svgWidth = state.timelineSvg.attr("width");
  if (state.rangeZoom && selectWidth > 10) {
    var x = select_block.attr("x");
    state.zoomHistory.push({zoom: prevZoom, start: $("#timeline").scrollLeft()});
    adjustZoom(svgWidth / selectWidth, false);
    $("#timeline").scrollLeft(x / prevZoom * state.zoom);
  }
  state.timelineSvg.selectAll("rect.select-block").remove();
  state.timelineSvg.selectAll("text.select-block").remove();
  mouseX = 0;
  state.timelineSvg.on("mousemove", null);
  state.timelineSvg.on("mousemove", mouseMoveHandlerWhenUp);
  $(document).on("keydown", defaultKeydown);
}

function turnOffMouseHandlers() {
  state.timelineSvg.on("mousedown", null);
  state.timelineSvg.on("mouseup", null);
  state.timelineSvg.on("mousemove", null);
}

function turnOnMouseHandlers() {
  state.timelineSvg.on("mousedown", mouseDownHandler);
  state.timelineSvg.on("mouseup", mouseUpHandler);
  state.timelineSvg.on("mousemove", mouseMoveHandlerWhenUp);
}

function drawLoaderIcon() {
  var loaderGroup = state.loaderSvg.append("g")
    .attr({
        id: "loader-icon",
    });
  loaderGroup.append("path")
    .attr({
      opacity: 0.2,
      fill: "#000",
      d: "M20.201,5.169c-8.254,0-14.946,6.692-14.946,14.946c0,8.255,6.692,14.946,14.946,14.946s14.946-6.691,14.946-14.946C35.146,11.861,28.455,5.169,20.201,5.169z M20.201,31.749c-6.425,0-11.634-5.208-11.634-11.634c0-6.425,5.209-11.634,11.634-11.634c6.425,0,11.633,5.209,11.633,11.634C31.834,26.541,26.626,31.749,20.201,31.749z"
    });
  var path = loaderGroup.append("path")
    .attr({
      fill: "#000",
      d: "M26.013,10.047l1.654-2.866c-2.198-1.272-4.743-2.012-7.466-2.012h0v3.312h0C22.32,8.481,24.301,9.057,26.013,10.047z"
    });
  path.append("animateTransform")
    .attr({
      attributeType: "xml",
      attributeName: "transform",
      type: "rotate",
      from: "0 20 20",
      to: "360 20 20",
      dur: "0.5s",
      repeatCount: "indefinite"
    });
}

function showLoaderIcon() {
  state.loaderSvg.select("g").attr("visibility", "visible");
}

function hideLoaderIcon() {
  state.loaderSvg.select("g").attr("visibility", "hidden");
}

function getMouseOver() {
  var paneWidth = $("#timeline").width();
  var left = paneWidth / 3;
  var right = paneWidth * 2 / 3;
  return function(d, i) {
    var p = d3.mouse(this);
    var x = parseFloat(p[0]);
    var relativeX = (x - $("#timeline").scrollLeft())
    var anchor = relativeX < left ? "start" :
                 relativeX < right ? "middle" : "end";
    var descView = state.timelineSvg.append("g").attr("id", "desc");

    var total = d.end - d.start;
    var initiation = "";
    if ((d.initiation != undefined) && d.initiation != "") {
      initiation = " initiated by " + state.operations[d.initiation];
    } 
    var title = d.title + initiation + ", total=" + total + "us, start=" + 
                d.start + "us, end=" + d.end+ "us";

    var text = descView.append("text")
      .attr("x", x)
      .attr("y", d.level * state.thickness - 5)
      .attr("text-anchor", anchor)
      .attr("class", "desc")
      .text(unescape(escape(title)));

		var bbox = text.node().getBBox();
		var padding = 2;
		var rect = descView.insert("rect", "text")
				.attr("x", bbox.x - padding)
				.attr("y", bbox.y - padding)
				.attr("width", bbox.width + (padding*2))
				.attr("height", bbox.height + (padding*2))
				.style("fill", "#222")
				.style("opacity", "0.7");
  };
}

var sizeHistory = 10;
var currentPos;
var nextPos;
var searchRegex = null;

function drawTimeline() {
  showLoaderIcon()
  updateURL(state.zoom, state.scale);
  var timelineGroup = state.timelineSvg.select("g#timeline");
  timelineGroup.selectAll("rect").remove();
  var timeline = timelineGroup.selectAll("rect")
    .data(state.dataToDraw, function(d) { return d.proc + "-" + d.id; });
  var mouseOver = getMouseOver();

  timeline.enter().append("rect");

  timeline
    .attr("id", function(d) { return "block-" + d.proc + "-" + d.id; })
    .attr("x", function(d) { return convertToPos(state, d.start); })
    .attr("y", function(d) { return d.level * state.thickness; })
    .style("fill", function(d) {
      if (!state.searchEnabled ||
          searchRegex[currentPos].exec(d.title) == null)
        return d.color;
      else return "#ff0000";
    })
    .attr("width", function(d) {
      return Math.max(constants.min_feature_width, convertToPos(state, d.end - d.start));
    })
    .attr("stroke", "black")
    .attr("stroke-width", "0.5")
    .attr("height", state.thickness)
    .style("opacity", function(d) {
      if (!state.searchEnabled ||
          searchRegex[currentPos].exec(d.title) != null)
        return d.opacity;
      else return 0.05;
    })
    .on("mouseout", function(d, i) { 
      state.timelineSvg.selectAll("#desc").remove();
    });

  timeline.on("mouseover", mouseOver);
  timeline.exit().remove();
  hideLoaderIcon();
}

function redraw() {
  filterAndMergeBlocks(state);
  drawTimeline();

  // TODO: put these in state
  var processorSvg = d3.select("#processors").select("svg");
  var names = processorSvg.selectAll("text");

  names.
    attr("y", levelCalculator)
    .style("opacity", function(proc) {
      return (proc.enabled) ? 1 : 0.5;
    });

  var timelineGroup = state.timelineSvg.select("g#timeline");
  var lines = state.timelineSvg.select("g#lines").selectAll("line");

  var thickness = state.thickness;
  lines
    .attr("y1", levelCalculator)
    .attr("y2", levelCalculator)
    .style("opacity", function(proc) {
      return (proc.enabled) ? 1 : 0.5;
    });
}

// This handler is called when you collapse/uncollapse a row in the
// timeline
function collapseHandler(d) {
  if (d.enabled) {
    for (var i = d.i + 1; i < state.processors.length; i++) {
      state.processors[i].base -= d.height;
    }
  } else {
    for (var i = d.i + 1; i < state.processors.length; i++) {
      state.processors[i].base += d.height;
    }
  }

  d.enabled = !d.enabled;
  if (!d.loaded) {
    showLoaderIcon();
    load_proc_timeline(d); // will redraw the timeline
    d.loaded = true;
  } else {
    // redraw the timeline
    redraw();
  }
}

function levelCalculator(proc) {
  var level = proc.base;
  if (proc.enabled) {
    level += proc.height;
  } else {
  }
  return level * state.thickness + state.thickness;
};

function drawProcessors() {

  var data = state.processors;
  var svg = d3.select("#processors").append("svg")
    .attr("width", constants.margin_left)
    .attr("height", state.height);

  var names = svg.selectAll(".processors")
    .data(data)
    .enter().append("text");

  var thickness = state.thickness;
  names.attr("text-anchor", "start")
    .attr("class", "processor")
    .attr("x", 0)
    .attr("y", levelCalculator)
    .style("fill", "#000000")
    .style("opacity", 0.5);

  names.each(function(d) {
    var text = d3.select(this);
    var tokens = d.processor.split(" to ");
    if (tokens.length == 1)
      text.append("tspan").text(d.processor);
    else {
      var source = tokens[0];
      var target = tokens[1].replace(" Channel", "");
      text.append("tspan").text(source)
        .attr({x : 0, dy : -10});
      text.append("tspan").text("==> " + target)
        .attr({x : 0, dy : 10});
    }
    text.on({
      "mouseover": function(d) {
        d3.select(this).style("cursor", "pointer")
      },
      "mouseout": function(d) {
        d3.select(this).style("cursor", "default")
      },
      "click": collapseHandler
    });
  });

  var lines = state.timelineSvg
    .append("g")
    .attr("id", "lines");

  var thickness = state.thickness;
  lines.selectAll(".lines")
    .data(data)
    .enter().append("line")
    .attr("x1", 0)
    .attr("y1", levelCalculator)
    .attr("x2", state.zoom * state.width)
    .attr("y2", levelCalculator)
    .style("stroke", "#000000")
    .style("stroke-width", "1px")
    .style("opacity", 0.5);
}

function drawHelpBox() {
  var paneWidth = $("#timeline").width();
  var paneHeight = $("#timeline").height();
  var helpBoxGroup = state.timelineSvg.append("g").attr("class", "help-box");
  var helpBoxWidth = Math.min(450, paneWidth - 100);
  var helpTextOffset = 20;
  var helpBoxHeight = Math.min(helpMessage.length * helpTextOffset + 100,
                               paneHeight - 100);

  var timelineWidth = state.timelineSvg.select("g#timeline").attr("width");
  var timelineHeight = state.timelineSvg.select("g#timeline").attr("height");
  var scrollLeft = $("#timeline").scrollLeft();
  var scrollTop = $(window).scrollTop();

  var thicknessRatio = state.thickness / state.baseThickness;
  var boxStartX = scrollLeft + (paneWidth - helpBoxWidth) / 2;
  var boxStartY = scrollTop + (paneHeight - helpBoxHeight) / 2 / (thicknessRatio);

  helpBoxGroup.append("rect")
    .attr({
        rx: 30,
        ry: 30,
        x: boxStartX,
        y: boxStartY,
        width: helpBoxWidth,
        height: helpBoxHeight,
        style: "fill: #222; opacity: 0.8;"
    });
  var helpText = helpBoxGroup.append("text")
    .attr("class", "help-box")
    .style("width", helpBoxWidth);
  var helpTitle = "Keyboard Shortcuts";
  helpText.append("tspan")
    .attr({ x: boxStartX + helpBoxWidth / 2, y: boxStartY + 50})
    .attr("text-anchor", "middle")
    .style("font-size", "20pt")
    .text(helpTitle);
  var off = 15;
  for (var i = 0; i < helpMessage.length; ++i) {
    helpText.append("tspan")
      .style("font-size", "12pt")
      .attr("text-anchor", "start")
      .attr({ x: boxStartX + 30, dy: off + helpTextOffset})
      .text(helpMessage[i]);
    off = 0;
  }
}

function drawSearchBox() {
  var paneWidth = $("#timeline").width();
  var paneHeight = $("#timeline").height();
  var searchBoxGroup = state.timelineSvg.append("g").attr("class", "search-box");
  var searchBoxWidth = Math.min(450, paneWidth - 100);
  var searchBoxHeight = Math.min(250, paneHeight - 100);

  var timelineWidth = state.timelineSvg.select("g#timeline").attr("width");
  var timelineHeight = state.timelineSvg.select("g#timeline").attr("height");
  var scrollLeft = $("#timeline").scrollLeft();
  var scrollTop = $(window).scrollTop();

  var thicknessRatio = state.thickness / state.baseThickness;
  var boxStartX = scrollLeft + (paneWidth - searchBoxWidth) / 2;
  var boxStartY = scrollTop + (paneHeight - searchBoxHeight) / 2 / thicknessRatio;

  searchBoxGroup.append("rect")
    .attr({
        rx: 30,
        ry: 30,
        x: boxStartX,
        y: boxStartY,
        width: searchBoxWidth,
        height: searchBoxHeight,
        style: "fill: #222; opacity: 0.8;"
    });
  var searchText = searchBoxGroup.append("text")
    .attr("class", "search-box")
    .style("width", searchBoxWidth);
  searchText.append("tspan")
    .attr({ x: boxStartX + searchBoxWidth / 2, y: boxStartY + 50})
    .attr("text-anchor", "middle")
    .style("font-size", "20pt")
    .text("Search");
  var searchInputWidth = searchBoxWidth - 40;
  var searchInputHeight = 50;
  searchBoxGroup.append("foreignObject")
    .attr({ x: boxStartX + 20, y: boxStartY + 150,
            width: searchInputWidth, height: searchInputHeight})
    .attr("text-anchor", "middle")
    .html("<input type='text' class='search-box' style='height: "
        + searchInputHeight + "px; width: "
        + searchInputWidth + "px; font-size: 20pt; font-family: Consolas, monospace;'></input>");
  $("input.search-box").focus();
}

function drawSearchHistoryBox() {
  var paneWidth = $("#timeline").width();
  var paneHeight = $("#timeline").height();
  var historyBoxGroup = state.timelineSvg.append("g").attr("class", "history-box");
  var historyBoxWidth = Math.min(450, paneWidth - 100);
  var historyBoxHeight = Math.min(350, paneHeight - 100);

  var timelineWidth = state.timelineSvg.select("g#timeline").attr("width");
  var timelineHeight = state.timelineSvg.select("g#timeline").attr("height");
  var scrollLeft = $("#timeline").scrollLeft();
  var scrollTop = $(window).scrollTop();

  var thicknessRatio = state.thickness / state.baseThickness;
  var boxStartX = scrollLeft + (paneWidth - historyBoxWidth) / 2;
  var boxStartY = scrollTop + (paneHeight - historyBoxHeight) / 2 / thicknessRatio;

  historyBoxGroup.append("rect")
    .attr({
        rx: 30,
        ry: 30,
        x: boxStartX,
        y: boxStartY,
        width: historyBoxWidth,
        height: historyBoxHeight,
        style: "fill: #222; opacity: 0.8;"
    });
  var historyText = historyBoxGroup.append("text")
    .attr("class", "history-box")
    .style("width", historyBoxWidth);
  var historyTitle = "Search History";
  historyText.append("tspan")
    .attr({ x: boxStartX + historyBoxWidth / 2, y: boxStartY + 50})
    .attr("text-anchor", "middle")
    .style("font-size", "20pt")
    .text(historyTitle);
  if (searchRegex != null) {
    var off = 15;
    var id = 1;
    for (var i = 0; i < sizeHistory; ++i) {
      var pos = (nextPos + i) % sizeHistory;
      var regex = searchRegex[pos];
      if (regex != null) {
        if (pos == currentPos) prefix = ">>> ";
        else prefix = id + " : ";
        historyText.append("tspan")
          .attr("text-anchor", "start")
          .attr({ x: boxStartX + 30, dy: off + 25})
          .text(prefix + regex.source);
        off = 0;
        id++;
      }
    }
  }
}

function updateURL() {
  var windowStart = $("#timeline").scrollLeft();
  var windowEnd = windowStart + $("#timeline").width();
  var start_time = convertToTime(state, windowStart);
  var end_time = convertToTime(state, windowEnd);
  var url = window.location.href.split('?')[0];
  url += "?start=" + start_time;
  url += "&end=" + end_time;
  url += "&collapseAll=" + state.collapseAll;
  url += "&resolution=" + state.resolution;
  if (state.searchEnabled)
    url += "&search=" + searchRegex[currentPos].source;
  window.history.replaceState("", "", url);
}

function adjustZoom(newZoom, scroll) {
  var prevZoom = state.zoom;
  state.zoom = Math.round(newZoom * 10) / 10;

  var svg = d3.select("#timeline").select("svg");

  svg.attr("width", state.zoom * state.width)
     .attr("height", state.height);

  var timelineGroup = svg.select("g#timeline");
  //timelineGroup.selectAll("rect").remove();

  svg.select("g#lines").selectAll("line")
    .attr("x2", state.zoom * state.width);
  svg.selectAll("#desc").remove();
  svg.selectAll("g.locator").remove();

  if (scroll) {
    var paneWidth = $("#timeline").width();
    var pos = ($("#timeline").scrollLeft() + paneWidth / 2) / prevZoom;
    // this will trigger a scroll event which in turn redraws the timeline
    $("#timeline").scrollLeft(pos * state.zoom - state.width / 2);
  } else {
    filterAndMergeBlocks(state);
    drawTimeline();
  }
}

function adjustThickness(newThickness) {
  state.thickness = newThickness;
  state.height = state.thickness * constants.max_level;
  d3.select("#processors").select("svg").remove();
  var svg = d3.select("#timeline").select("svg");
  var lines = state.timelineSvg.select("g#lines");
  lines.remove();

  svg.attr("width", state.zoom * state.width)
     .attr("height", state.height);
  //var filteredData = filterAndMergeBlocks(profilingData, zoom, scale);
  drawTimeline();
  drawProcessors();
  svg.selectAll("#desc").remove();
  svg.selectAll("g.locator").remove();
}

function suppressdefault(e) {
  if (e.preventDefault) e.preventDefault();
  if (e.stopPropagation) e.stopPropagation();
}

function setKeyHandler(handler) {
  $(document).off("keydown");
  $(document).on("keydown", handler);
}

function makeModalKeyHandler(validKeys, callback) {
  return function(e) {
    if (!e) e = event;
    var code = e.keyCode || e.charCode;
    if (!(e.ctrlKey || e.metaKey || e.altKey)) {
      for (var i = 0; i < validKeys.length; ++i) {
        if (keys[code] == validKeys[i]) {
          callback(keys[code]);
          return false;
        }
      }
    }
    return true;
  }
}

function defaultKeydown(e) {
  if (!e) e = event;

  var code = e.keyCode || e.charCode;
  var commandType = Command.none;
  var modifier = e.ctrlKey || e.metaKey || e.altKey;
  var multiFnKeys = e.metaKey && e.ctrlKey || e.altKey && e.ctrlKey;

  var translatedCode = keys[code];

  if (!modifier) {
    commandType = noModifierCommands[translatedCode];
  } else if (!multiFnKeys) {
    commandType = modifierCommands[translatedCode];
  } else {
    commandType = multipleModifierCommands[translatedCode];
  }

  if (commandType == undefined) {
    if (e.metaKey || e.altKey)
      state.rangeZoom = false;
    return true;
  }

  suppressdefault(e);
  if (commandType == Command.help) {
    turnOffMouseHandlers();
    makeTimelineTransparent();
    drawHelpBox();
    setKeyHandler(makeModalKeyHandler(['/', 'esc'], function(key) {
      state.timelineSvg.select("g.help-box").remove();
      makeTimelineOpaque();
      setKeyHandler(defaultKeydown);
      turnOnMouseHandlers();
    }));
    return false;
  }
  else if (commandType == Command.search) {
    turnOffMouseHandlers();
    makeTimelineTransparent();
    drawSearchBox();
    setKeyHandler(makeModalKeyHandler(['enter', 'esc'], function(key) {
      if (key == 'enter') {
        var re = $("input.search-box").val();
        //console.log("Search Expression: " + re);
        if (re.trim() != "") {
          if (searchRegex == null) {
            searchRegex = new Array(sizeHistory);
            currentPos = -1;
            nextPos = 0;
          }
          currentPos = nextPos;
          nextPos = (nextPos + 1) % sizeHistory;
          searchRegex[currentPos] = new RegExp(re);
          state.searchEnabled = true;
        }
      }
      state.timelineSvg.select("g.search-box").remove();
      if (state.searchEnabled) {
        filterAndMergeBlocks(state);
        drawTimeline();
      }
      makeTimelineOpaque();
      setKeyHandler(defaultKeydown);
      turnOnMouseHandlers();
    }));
    return false;
  }
  else if (commandType == Command.search_history) {
    turnOffMouseHandlers();
    makeTimelineTransparent();
    drawSearchHistoryBox();
    setKeyHandler(makeModalKeyHandler(['h', 'esc'], function(key) {
      state.timelineSvg.select("g.history-box").remove();
      makeTimelineOpaque();
      setKeyHandler(defaultKeydown);
      turnOnMouseHandlers();
    }));
    return false;
  }

  if (commandType == Command.zix) {
    var inc = 4.0;
    adjustZoom(state.zoom + inc, true);
  } else if (commandType == Command.zox) {
    var dec = 4.0;
    if (state.zoom - dec > 0)
      adjustZoom(state.zoom - dec, true);
  } else if (commandType == Command.zrx) {
    state.zoomHistory = Array();
    adjustZoom(1.0, false);
  } else if (commandType == Command.zux) {
    if (state.zoomHistory.length > 0) {
      var previousZoomHistory = state.zoomHistory.pop();
      adjustZoom(previousZoomHistory.zoom, false);
      $("#timeline").scrollLeft(previousZoomHistory.start);
    }
  } else if (commandType == Command.ziy)
    adjustThickness(state.thickness * 2);
  else if (commandType == Command.zoy)
    adjustThickness(state.thickness / 2);
  else if (commandType == Command.zry) {
    state.height = $(window).height() - margin_bottom;
    state.thickness = state.height / constants.max_level;
    adjustThickness(state.thickness);
  }
  else if (commandType == Command.clear_search) {
    state.searchEnabled = false;
    searchRegex = null;
    filterAndMergeBlocks(state);
    drawTimeline();
  }
  else if (commandType == Command.toggle_search) {
    if (searchRegex != null) {
      state.searchEnabled = !state.searchEnabled;
      filterAndMergeBlocks(state);
      drawTimeline();
    }
  }
  else if (commandType == Command.previous_search ||
           commandType == Command.next_search) {
    if (state.searchEnabled) {
      var pos = commandType == Command.previous_search ?
                (currentPos - 1 + sizeHistory) % sizeHistory :
                (currentPos + 1) % sizeHistory;
      var sentinel = commandType == Command.previous_search ?
                (nextPos - 1 + sizeHistory) % sizeHistory : nextPos;
      if (pos != sentinel && searchRegex[pos] != null) {
        currentPos = pos;
        filterAndMergeBlocks(state);
        drawTimeline();
      }
    }
  }
  return false;
}

function defaultKeyUp(e) {
  if (!e) e = event;
  if (!(e.metaKey || e.altKey)) {
    state.rangeZoom = true;
  }
  return true;
}

function load_proc_timeline(proc) {
  var proc_name = proc.processor;
  state.processorData[proc_name] = {};
  d3.tsv(proc.tsv,
    function(d, i) {
        var level = +d.level;
        var start = +d.start;
        var end = +d.end;
        var total = end - start;
        if (total > state.resolution) {
            return {
                id: i,
                level: level,
                start: start,
                end: end,
                color: d.color,
                opacity: d.opacity,
                initiation: d.initiation,
                title: d.title
            };
        }
    },
    function(data) {
      // split profiling items by which level they're on
      for(var i = 0; i < data.length; i++) {
        var d = data[i];
        if (d.level in state.processorData[proc_name]) {
          state.processorData[proc_name][d.level].push(d);
        } else {
          state.processorData[proc_name][d.level] = [d];
        }
      }
      redraw();
    }
  );
}

function initTimelineElements() {
  var timelineGroup = state.timelineSvg.append("g")
      .attr("id", "timeline");

  $("#timeline").scrollLeft(0);
  parseURLParameters();

  var windowCenterY = $(window).height() / 2;
  $(window).scroll(function() {
      $("#loader-icon").css("top", $(window).scrollTop() + windowCenterY);
  });

  // set scroll callback
  var timer = null;
  $("#timeline").scroll(function() {
      if (timer !== null) {
        clearTimeout(timer);
      }
      timer = setTimeout(function() {
        filterAndMergeBlocks(state);
        drawTimeline();
      }, 100);
  });
  if (!state.collapseAll) {
    // initially load in all the cpu processors
    state.processors.forEach(function(proc) {
      var cpu = /CPU Processor/;
      if (cpu.test(proc.processor)) {
        collapseHandler(proc);
      }
    });
  }
  turnOnMouseHandlers();
}


function load_ops_and_timeline() {
  d3.tsv("legion_prof_ops.tsv", 
    function(d) {
      return d;
    },
    function(data) { 
      data.forEach(function(d) {
        state.operations[parseInt(d.op_id)] = d.operation;      
      });
      // initTimelineElements depends on the ops to be loaded
      initTimelineElements();
    }
  );
}

function load_procs(callback) {
  d3.tsv('legion_prof_processor.tsv',
    function(d) {
      return {
        processor: d.processor,
        height: +d.levels,
        tsv: d.tsv
      };
    },
    function(data) {
      var prevEnd = 0;
      var base = 0;
      var i = 0;
      data.forEach(function(proc) {
        proc.enabled = false;
        proc.loaded = false;
        proc.base = base;
        proc.i = i;
        base += 2;
        i+= 1;
      });
      //console.log(data);
      state.processors = data;
      drawProcessors();
      callback();
    }
  );
}

function load_data() {
  load_procs(load_ops_and_timeline);
}

function initializeState() {
  var margin = constants.margin_left + constants.margin_right;
  state.width = $(window).width() - margin;
  state.height = $(window).height() - constants.margin_bottom;
  state.scale = state.width / (constants.end - constants.start);
  state.zoom = 1.0;
  state.zoomHistory = Array();
  state.profilingData = {};
  state.processorData = {};
  state.operations = {};
  state.thickness = Math.max(state.height / constants.max_level, 20);
  state.baseThickness = state.height / constants.max_level;
  state.height = constants.max_level * state.thickness;
  state.resolution = 10; // time (in us) of the smallest feature we want to load
  state.searchEnabled = false;
  state.rangeZoom = true;
  state.collapseAll = false;

  setKeyHandler(defaultKeydown);
  $(document).on("keyup", defaultKeyUp);

  state.timelineSvg = d3.select("#timeline").append("svg")
    .attr("width", state.zoom * state.width)
    .attr("height", state.height);

  state.loaderSvg = d3.select("#loader-icon").append("svg")
    .attr("width", "40px")
    .attr("height", "40px");

  drawLoaderIcon();
  load_data();
}

function main() {
  $.getJSON("json/scale.json", function(json) {
    $.each(json, function(key, val) {
      constants[key] = val;
    });
  }).done(initializeState);
  // TODO: Error messages
}

main();
