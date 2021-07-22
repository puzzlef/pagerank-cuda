const fs = require('fs');
const os = require('os');
const path = require('path');

const RGRAPH = /^Loading graph .*\/(.+?)\.mtx \.\.\./m;
const RORDER = /^order: (\d+) size: (\d+) \{\}$/m;
const RRESLT = /^\[(.+?) ms; (.+?) iters\.\] \[(.+?) err\.\] (\w+)(?:<<<(\d+), (\d+)>>>)?/m;




// *-FILE
// ------

function readFile(pth) {
  var d = fs.readFileSync(pth, 'utf8');
  return d.replace(/\r?\n/g, '\n');
}

function writeFile(pth, d) {
  d = d.replace(/\r?\n/g, os.EOL);
  fs.writeFileSync(pth, d);
}




// *-CSV
// -----

function writeCsv(pth, rows) {
  var cols = Object.keys(rows[0]);
  var a = cols.join()+'\n';
  for (var r of rows)
    a += [...Object.values(r)].map(v => `"${v}"`).join()+'\n';
  writeFile(pth, a);
}




// *-LOG
// -----

function readLogLine(ln, data, state) {
  if (RGRAPH.test(ln)) {
    var [, graph] = RGRAPH.exec(ln);
    if (!data.has(graph)) data.set(graph, []);
    state = {graph};
  }
  else if (RORDER.test(ln)) {
    var [, order, size] = RORDER.exec(ln);
    state.order = parseFloat(order);
    state.size  = parseFloat(size);
  }
  else if (RRESLT.test(ln)) {
    var [, time, iterations, error, technique, blocks, threads] = RRESLT.exec(ln);
    data.get(state.graph).push(Object.assign({}, state, {
      time:       parseFloat(time),
      iterations: parseFloat(iterations),
      error:      parseFloat(error),
      technique,
      blocks:     parseFloat(blocks||'0'),
      threads:    parseFloat(threads||'0')
    }));
  }
  return state;
}

function readLog(pth) {
  var text = readFile(pth);
  var lines = text.split('\n');
  var data = new Map();
  var state = null;
  for (var ln of lines)
    state = readLogLine(ln, data, state);
  return data;
}




// PROCESS-*
// ---------

function processShortGraph(rows) {
  var a = [];
  var techniques = new Set(rows.map(r => r.technique));
  for (var technique of techniques) {
    var frows = rows.filter(r => r.technique===technique);
    var i = frows.reduce((m, v, i, x) => v.time < x[m].time? i:m, 0);
    a.push(frows[i]);
  }
  return a;
}


function processCsv(data) {
  var a = [];
  for (var rows of data.values())
    a.push(...rows);
  return a;
}


function processShortCsv(data) {
  var a = [];
  for (var rows of data.values())
    a.push(...processShortGraph(rows));
  return a;
}


function processShortLog(data) {
  var a = '';
  for (var rows of data.values()) {
    var frows = processShortGraph(rows), r = frows[0];
    a += `Loading graph ${r.graph}.mtx ...\n`;
    a += `order: ${r.order} size: ${r.size} {}\n`;
    for (var r of frows) {
      var time       = r.time.toFixed(3).padStart(9, '0');
      var iterations = r.iterations.toFixed(0).padStart(3, '0');
      var error      = r.error.toExponential(4);
      var technique  = r.blocks? `${r.technique}<<<${r.blocks}, ${r.threads}>>>` : r.technique;
      a += `[${time} ms; ${iterations} iters.] [${error} err.] ${technique}\n`;
    }
    a += '\n';
  }
  return a.trim()+'\n';
}



// MAIN
// ----

function main(cmd, log, out) {
  var data = readLog(log);
  if (path.extname(out)==='') cmd += '-dir';
  switch (cmd) {
    case 'csv':
      var rows = processCsv(data);
      writeCsv(out, rows);
      break;
    case 'csv-dir':
      for (var [graph, rows] of data)
        writeCsv(path.join(out, graph+'.csv'), rows);
      break;
    case 'short-csv':
      var rows = processShortCsv(data);
      writeCsv(out, rows);
      break;
    case 'short-csv-dir':
      for (var [graph, rows] of data) {
        var rows = processShortCsv(new Map([[graph, rows]]));
        writeCsv(path.join(out, graph+'.short.csv'), rows);
      }
      break;
    case 'short-log':
      var text = processShortLog(data);
      writeFile(out, text);
      break;
    case 'short-log-dir':
      for (var [graph, rows] of data) {
        var text = processShortLog(data);
        writeFile(path.join(out, graph+'.short.txt'), text);
      }
      break;
    default:
      console.error(`error: "${cmd}"?`);
      break;
  }
}
main(...process.argv.slice(2));
