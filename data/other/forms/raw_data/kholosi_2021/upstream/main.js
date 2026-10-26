function kde(kernel, thresholds, data) {
    return thresholds.map(t => [t, d3.mean(data, d => kernel(t - d))]);
}

function epanechnikov(bandwidth) {
    return x => Math.abs(x /= bandwidth) <= 1 ? 0.75 * (1 - x * x) / bandwidth : 0;
}

function leastSquares(xSeries, ySeries) {
    var reduceSumFunc = function(prev, cur) { return prev + cur; };
    
    var xBar = xSeries.reduce(reduceSumFunc) * 1.0 / xSeries.length;
    var yBar = ySeries.reduce(reduceSumFunc) * 1.0 / ySeries.length;

    var ssXX = xSeries.map(function(d) { return Math.pow(d - xBar, 2); })
        .reduce(reduceSumFunc);
    
    var ssYY = ySeries.map(function(d) { return Math.pow(d - yBar, 2); })
        .reduce(reduceSumFunc);
        
    var ssXY = xSeries.map(function(d, i) { return (d - xBar) * (ySeries[i] - yBar); })
        .reduce(reduceSumFunc);
        
    var slope = ssXY / ssXX;
    var intercept = yBar - (xBar * slope);
    var rSquare = Math.pow(ssXY, 2) / (ssXX * ssYY);
    
    return [slope, intercept, rSquare];
}

var margin = {top: 50, right: 50, bottom: 50, left: 50}
var width = window.innerWidth - margin.right - margin.left,
    height = window.innerHeight - margin.top - margin.bottom;
var container = d3.select("#map")
    .attr("width", width + margin.right + margin.left)
    .attr("height", height + margin.top + margin.bottom)

var svg = container.append("svg")
    .attr("width", width + margin.right + margin.left)
    .attr("height", height + margin.top + margin.bottom)
    .append("g")
        .attr("transform",
        "translate(" + margin.left + "," + margin.top + ")")

var x = d3.scaleLinear()
    .domain([3000, 400])
    .range([0, width]);
// var x = d3.scaleLinear()
//     .domain([3600, 2000])
//     .range([0, width]);
svg.append("g")
    .call(d3.axisTop(x));

var y = d3.scaleLinear()
    .domain([1100, 250])
    .range([height, 0]);
svg.append("g")
    .attr("transform", `translate(${width}, 0)`)
    .call(d3.axisRight(y));
svg.append("text")
    .attr("text-anchor", "start")
    .attr("x", 0)
    .attr("y", 20)
    .text("F2 (Hz)");
svg.append("text")
    .attr("class", "y label")
    .attr("text-anchor", "start")
    .attr("y", width - 20)
    .attr("x", -height)
    .attr("dy", ".75em")
    .attr("transform", "rotate(-90)")
    .text("F1 (Hz)");

var t = d3.transition()
    .duration(750)

vowels = [
    // ['i', 240, 2400],
    // ['y', 235, 2100],
    // ['e', 390, 2300],
    // ['ø', 370, 1900],
    // ['ɛ', 610, 1900],
    // ['œ', 585, 1710],
    // ['a', 850, 1610],
    // ['ɶ', 820, 1530],
    // ['ɑ', 750, 940],
    // ['ɒ', 700, 760],
    // ['ʌ', 600, 1170],
    // ['ɔ', 500, 700],
    // ['ɤ', 460, 1310],
    // ['o', 360, 640],
    // ['ɯ', 300, 1390],
    // ['u', 250, 595]
]

vowels.forEach(d => {
    svg.append("text")
        .text(d[0])
        .attr("x", x(d[2]))
        .attr("y", y(d[1]))
        .style("font-size", 30)
        .style("font-family", "Noto Sans")
})

var stringToColour = function(str) {
    str += "jhajhf"
    try {
        var hash = 0
        for (var i = 0; i < str.length; i++) {
            hash = str.charCodeAt(i) + ((hash << 4) - hash)
        }
        var colour = '#'
        for (var i = 0; i < 3; i++) {
            var value = (hash >> (i * 8)) & 0xFF
            colour += ('00' + value.toString(16)).substr(-2)
        }
        return colour
    }
    catch {
        return "0"
    }
}

d3.tsv("data.txt").then(function(data) {
    var sums = {}, counts = {}
    var g = svg.append("g")
    var g2 = svg.append("g")
    var res = []
    var cur = {}
    for (var i = 0; i < data.length; i++) {
        if (data[i].environment != "") {
            if (i != 0) res.push(cur)
            cur = {'env': data[i].environment, 'data': []}
        }
        else {
            cur.data.push(data[i])
        }
    }
    res.push(cur)
    f1s = {}
    f2s = {}
    lens = {}
    res.forEach((elem, i) => {
        var len = elem.data[elem.data.length - 1].Time_s - elem.data[0].Time_s
        var word = elem.env
        // if (word[word.length - 1] = '_') return
        var old_word = word
        var style = word.replace(/_(.*?)_/g, '<span style="color: blue">$1</span>')
        var vowel = word.match(/_.*?_/g)[0]
        if (!counts[vowel]) counts[vowel] = 0, sums[vowel] = [0, 0]

        if (!(vowel in f1s)) f1s[vowel] = []
        if (!(vowel in f2s)) f2s[vowel] = []
        if (!(vowel in lens)) lens[vowel] = []
        f1s[vowel].push(+elem.data[Math.floor(elem.data.length / 2)].F1_Hz)
        f2s[vowel].push(+elem.data[Math.floor(elem.data.length / 2)].F2_Hz)
        lens[vowel].push(Math.round(len * 1000))

        counts[vowel]++
        sums[vowel][0] += +elem.data[Math.floor(elem.data.length / 2)].F1_Hz
        sums[vowel][1] += +elem.data[Math.floor(elem.data.length / 2)].F2_Hz
        word = word.replace(/_(.*?)_/g, '$1')
        // if (vowel != '_a_' && vowel != '_o_') return
        function mouseover() {
            var last = null
            d3.selectAll("circle").style("opacity", 0.1)
            d3.selectAll("line").style("opacity", 0.1)
            d3.selectAll("text").style("opacity", 0.6)
            elem.data.forEach((d, i) => {
                g.append("circle")
                    .attr("class", "trace data-" + word)
                    .attr("data-time", d.Time_s - elem.data[0].Time_s)
                    .attr("cx", x(d.F2_Hz))
                    .attr("cy", y(d.F1_Hz)).transition().duration(250)
                    .attr("r", 2)
                    .attr("fill", stringToColour(vowel))
                if (i != 0) {
                    g.append("line")
                        .attr("class", "trace data-" + word)
                        .attr("data-time", d.Time_s - elem.data[0].Time_s)
                        .attr("x1", x(last.F2_Hz))
                        .attr("y1", y(last.F1_Hz))
                        .attr("x2", x(last.F2_Hz))
                        .attr("y2", y(last.F1_Hz)).transition().duration(1000)
                        .attr("x2", x(d.F2_Hz))
                        .attr("y2", y(d.F1_Hz))
                        .attr("stroke-width", 1)
                        .attr("stroke", stringToColour(vowel))
                }
                last = d
            })
            d3.select("h1").html('/' + style + '/ (' + Math.round(len * 1000) + ' ms)')
            d3.select("p").html('F1: ' + elem.data[Math.floor(elem.data.length / 2)].F1_Hz + ', F2: ' +
                elem.data[Math.floor(elem.data.length / 2)].F2_Hz)
            d3.selectAll(".data-" + word).style("opacity", 1)
            var margin = {top: 10, right: 10, bottom: 50, left: 50},
                width = 260 - margin.left - margin.right,
                height = 200 - margin.top - margin.bottom;
            var svg2 = d3.select("div")
                .append("svg")
                .attr("class", "graph")
                .attr("width", width + margin.left + margin.right)
                .attr("height", height + margin.top + margin.bottom)
                .append("g")
                    .attr("transform",
                    "translate(" + margin.left + "," + margin.top + ")")
            var x2 = d3.scaleLinear()
                .domain([0, len])
                .range([ 0, width ])
            var y2 = d3.scaleLinear()
                .domain([0, 5000])
                .range([height, 0])
            svg2.append("g")
                .attr("transform", "translate(0," + height + ")")
                .call(d3.axisBottom(x2))
                .selectAll("text")
                    .attr("y", 0)
                    .attr("x", 9)
                    .attr("dy", ".35em")
                    .attr("transform", "rotate(90)")
                    .style("text-anchor", "start");
            svg2.append("g")
                .call(d3.axisLeft(y2))

            elem.data.forEach(d => {
                svg2.append("circle")
                    .attr("cx", x2(d.Time_s - elem.data[0].Time_s))
                    .attr("cy", y2(d.F1_Hz))
                    .attr("r", 1.5)
                    .style("fill", stringToColour("F1"))
                svg2.append("circle")
                    .attr("cx", x2(d.Time_s - elem.data[0].Time_s))
                    .attr("cy", y2(d.F2_Hz))
                    .attr("r", 1.5)
                    .style("fill", stringToColour("F2"))
                svg2.append("circle")
                    .attr("cx", x2(d.Time_s - elem.data[0].Time_s))
                    .attr("cy", y2(d.F3_Hz))
                    .attr("r", 1.5)
                    .style("fill", stringToColour("F3"))
                svg2.append("circle")
                    .attr("cx", x2(d.Time_s - elem.data[0].Time_s))
                    .attr("cy", y2(d.F4_Hz))
                    .attr("r", 1.5)
                    .style("fill", stringToColour("F4"))})
                svg2.append("text")
                    .attr("text-anchor", "center")
                    .attr("x", height / 2)
                    .attr("y", width - 15)
                    .text("Time (s)")
                    .style("font-size", "0.6em")
                    .style("fill", "white")
                svg2.append("text")
                    .attr("class", "y label")
                    .attr("text-anchor", "center")
                    .attr("y", -50)
                    .attr("x", -width / 2)
                    .attr("dy", ".75em")
                    .attr("transform", "rotate(-90)")
                    .text("Frequency (Hz)")
                    .style("font-size", "0.6em")
                    .style("fill", "white")
        }
        function mouseout() {
            d3.select("div")
                .style("opacity", 0)
            d3.selectAll("*").style("opacity", null)
            d3.selectAll(".graph, .trace")
                .remove()
            d3.select("h1").html("")
            d3.select("p").html("")
        }
        if (vowel.length != 3) {
            return
            g2.append("circle")
                .attr("class", "data-" + word)
                .attr("cx", x(elem.data[elem.data.length - 1].F2_Hz))
                .attr("cy", y(elem.data[elem.data.length - 1].F1_Hz))
                .attr("r", 10 * Math.sqrt(len / 0.2))
                // .attr("r", 5)
                .attr("fill", stringToColour(vowel))
                .attr("stroke", "black")
                .attr("stroke-width", 1)
                .attr("opacity", 0.8)
                .on("mouseover", mouseover)
                .on("mouseout", mouseout)
            g2.append("circle")
                .attr("class", "data-" + word)
                .attr("cx", x(elem.data[0].F2_Hz))
                .attr("cy", y(elem.data[0].F1_Hz))
                .attr("r", 10 * Math.sqrt(len / 0.2))
                // .attr("r", 5)
                .attr("fill", stringToColour(vowel))
                .attr("stroke", "black")
                .attr("stroke-width", 1)
                .attr("opacity", 0.8)
                .on("mouseover", mouseover)
                .on("mouseout", mouseout)
            g.append("line")
                .attr("class", "data-" + word)
                .attr("x1", x(elem.data[0].F2_Hz))
                .attr("y1", y(elem.data[0].F1_Hz))
                .attr("x2", x(elem.data[elem.data.length - 1].F2_Hz))
                .attr("y2", y(elem.data[elem.data.length - 1].F1_Hz))
                .attr("stroke-width", 1)
                .attr("stroke", stringToColour(vowel))
                .attr("opacity", 0.8)
        }
        else {
            g2.append("circle")
                .attr("class", "data-" + word)
                .attr("cx", x(elem.data[Math.floor(elem.data.length / 2)].F2_Hz))
                .attr("cy", y(elem.data[Math.floor(elem.data.length / 2)].F1_Hz))
                .attr("r", 10 * Math.sqrt(len / 0.2))
                // .attr("r", 5)
                .attr("fill", stringToColour(vowel))
                .attr("stroke", "black")
                .attr("stroke-width", 1)
                .attr("opacity", 0.7)
                .on("mouseover", mouseover)
                .on("mouseout", mouseout)
            // g.append("line")
            //     .attr("class", "data-" + word)
            //     .attr("x1", x(elem.data[Math.floor((elem.data.length / 10))].F2_Hz))
            //     .attr("y1", y(elem.data[Math.floor((elem.data.length / 10))].F1_Hz))
            //     .attr("x2", x(elem.data[Math.floor(elem.data.length / 2)].F2_Hz))
            //     .attr("y2", y(elem.data[Math.floor(elem.data.length / 2)].F1_Hz))
            //     .attr("stroke-width", 1)
            //     .attr("stroke", stringToColour(vowel))
            //     .attr("opacity", 0.8)
            // g2.append("text")
            //     .html(old_word)
            //     .attr("x", x(elem.data[Math.floor(elem.data.length / 2)].F2_Hz))
            //     .attr("y", y(elem.data[Math.floor(elem.data.length / 2)].F1_Hz))
            //     .attr("text-anchor", "middle")
        }
    })

    var hist = d3.select("#histogram")
        .attr("width", width + margin.left + margin.right)
        .attr("height", height + margin.top + margin.bottom)
        .append("g")
            .attr("transform", "translate(" + margin.left + "," + margin.top + ")")
    
    var x2 = d3.scaleLinear()
        .domain([0, 400])
        .range([0, width])

    hist.append("g")
        .attr("transform", "translate(0," + height + ")")
        .call(d3.axisBottom(x2))

    var y2 = d3.scaleLinear()
        .range([height, 0])
        .domain([0, 0.02]);
    hist.append("g")
        .call(d3.axisLeft(y2))

    var calc = d3.histogram()
        .thresholds(d3.range(0, x2.domain()[1] + 1, 20))
        .domain(x2.domain())

    for (const vowel in sums) {
        if (vowel.length != 3) continue
        // console.log(sums[vowel], counts[vowel])
        g2.append("text")
            .attr("x", x(sums[vowel][1] / counts[vowel]))
            .attr("y", y(sums[vowel][0] / counts[vowel]))
            .text("/" + vowel.slice(1, vowel.length - 1) + "/")
            .attr("font-size", 30)
            .style("pointer-events", "none")
        g2.append("text")
            .attr("x", x(sums[vowel][1] / counts[vowel]) + 40)
            .attr("y", y(sums[vowel][0] / counts[vowel]) - 12)
            .text('F1: ' + Math.round(d3.mean(f1s[vowel]) / 10) * 10 + '±' + Math.round(d3.deviation(f1s[vowel]) / 10) * 10 + ' Hz')
            .attr("font-size", 13)
            .style("pointer-events", "none")
        g2.append("text")
            .attr("x", x(sums[vowel][1] / counts[vowel]) + 40)
            .attr("y", y(sums[vowel][0] / counts[vowel]))
            .text('F2: ' + Math.round(d3.mean(f2s[vowel]) / 10) * 10 + '±' + Math.round(d3.deviation(f2s[vowel]) / 10) * 10 + ' Hz')
            .attr("font-size", 13)
            .style("pointer-events", "none")
        g2.append("text")
            .attr("x", x(sums[vowel][1] / counts[vowel]) + 40)
            .attr("y", y(sums[vowel][0] / counts[vowel]) + 12)
            .text('Len: ' + Math.round(d3.mean(lens[vowel])) + '±' + Math.round(d3.deviation(lens[vowel])) + ' ms')
            .attr("font-size", 13)
            .style("pointer-events", "none")
        g2.append("text")
            .attr("x", x(sums[vowel][1] / counts[vowel]) + 40)
            .attr("y", y(sums[vowel][0] / counts[vowel]) + 24)
            .text('Count: ' + counts[vowel])
            .attr("font-size", 13)
            .style("pointer-events", "none")

        // console.log(lens[vowel])
        bins = calc(lens[vowel])
        // console.log(bins)

        // if (vowel == '_ɔ_') hist.append("g")
        //     .attr("fill", stringToColour(vowel))
        //     .selectAll("rect")
        //     .data(bins)
        //     .join("rect")
        //         .attr("x", d => x2(d.x0) + 1)
        //         .attr("y", d => y2(d.length / lens[vowel].length))
        //         .attr("width", d => x2(d.x1) - x2(d.x0) - 1)
        //         .attr("height", d => y2(0) - y2(d.length / lens[vowel].length));

        density = kde(epanechnikov(20), d3.range(0, x2.domain()[1] + 1, 10), lens[vowel])
        // console.log(density)

        hist.append("path")
            .datum(density)
            .attr("fill", stringToColour(vowel))
            .attr("stroke", stringToColour(vowel))
            .attr("fill-opacity", 0.2)
            .attr("stroke-width", 5)
            .attr("stroke-linejoin", "round")
            .attr("d", d3.line()
                .curve(d3.curveBasis)
                .x(d => x2(d[0]))
                .y(d => y2(d[1])))
    }



    var hist = d3.select("#graph")
        .attr("width", width + margin.left + margin.right)
        .attr("height", height + margin.top + margin.bottom)
        .append("g")
            .attr("transform", "translate(" + margin.left + "," + margin.top + ")")
    
    var x2 = d3.scaleLinear()
        .domain([0, 3000])
        .range([0, width])

    hist.append("g")
        .attr("transform", "translate(0," + height + ")")
        .call(d3.axisBottom(x2))

    var y2 = d3.scaleLinear()
        .range([height, 0])
        .domain([0, 3000]);
    hist.append("g")
        .call(d3.axisLeft(y2))

    var calc = d3.histogram()
        .thresholds(d3.range(0, x2.domain()[1] + 1, 20))
        .domain(x2.domain())

    var locus = hist.append("g")
    res.forEach((elem, i) => {
        var c = ""
        if (elem.env.match(/(.ʰ?)?_.*?_/g)[0][0] == 't' || elem.env.match(/(.ʰ?)?_.*?_/g)[0][0] == 'd') c = 'alveolar'
        if (elem.env.match(/(.ʰ?)?_.*?_/g)[0][0] == 'k' || elem.env.match(/(.ʰ?)?_.*?_/g)[0][0] == 'g') c = 'velar'
        if (elem.env.match(/(.ʰ?)?_.*?_/g)[0][0] == 'p' || elem.env.match(/(.ʰ?)?_.*?_/g)[0][0] == 'b') c = 'labial'
        locus.append("circle")
            .attr("cx", x2(+elem.data[Math.floor(elem.data.length / 2)].F2_Hz))
            .attr("cy", y2(+elem.data[Math.floor(elem.data.length / 100)].F2_Hz))
            .attr("fill", (c && stringToColour(c)) || "grey")
            .attr("r", 5)
            .attr("fill-opacity", 0.8)
    })
})