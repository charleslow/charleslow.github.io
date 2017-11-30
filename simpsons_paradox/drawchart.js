function drawchart() {
  var number = 10,
      bar_width = 50,
      slider_position = 20,
      bar_position = 30,
      handle_position = 0,
      bar_class = "bar_r",
      handle_class = "handle_r",
      bar_label_class = "bar_label_r";

  function chart(selection) {
    selection.each(function() {
      var slider = d3.select(this).append("g")
        .attr("class", "slider")
        .attr("transform", "translate(" + slider_position + ",0)");

      slider.append("line")
        .attr("class", "track")
        .attr("y1", y.range()[0])
        .attr("y2", y.range()[1])
      .select(function() { return this.parentNode.appendChild(this.cloneNode(true)); })
        .attr("class", "track-overlay")
        .call(d3.drag()
            .on("start.interrupt", function() { slider.interrupt(); })
            .on("start drag", function() { change(50 - y.invert(d3.event.y)); }));

      var bar = d3.select(this).append("g")
        .append("rect")
        .attr("class", bar_class)
        .attr("x", bar_position)
        .attr("y", y(50 - number))
        .attr("width", bar_width)
        .attr("height", y(number));

      var handle = slider.insert("rect", ".track-overlay")
        .attr("class", handle_class)
        .attr("height", 2)
        .attr("width", bar_width + 10)
        .attr("x", handle_position)
        .attr("y", y(50 - number) - 2);

      var bar_label = d3.select(this).append("text")
        .attr("class", bar_label_class)
        .attr("x", bar_position + (bar_width / 2))
        .attr("y", y(50 - number) - 5)
        .text(number);

      function change(h) {
        var updated_number = Math.round(h);

        handle
          .attr("y", y(50 - updated_number) - 2);

        bar
          .attr("y", y(50 - updated_number))
          .attr("height", y(updated_number));

        bar_label
          .attr("y", y(50 - updated_number) - 5)
          .text(updated_number);

        chart.number = updated_number;

        // Change the combined A, B bar heights
        // Jar A
        a_r_bar
          .attr("height", y2(jar1_r_chart.number + jar2_r_chart.number))
          .attr("y", y2(100 - (jar1_r_chart.number + jar2_r_chart.number)));

        a_r_label
          .text(jar1_r_chart.number + jar2_r_chart.number)
          .attr("y", y2(100 - (jar1_r_chart.number + jar2_r_chart.number)) - 5);

        a_g_bar
          .attr("height", y2(jar1_g_chart.number + jar2_g_chart.number))
          .attr("y", y2(100 - (jar1_g_chart.number + jar2_g_chart.number)));

        a_g_label
          .text(jar1_g_chart.number + jar2_g_chart.number)
          .attr("y", y2(100 - (jar1_g_chart.number + jar2_g_chart.number)) - 5);

        // Jar B
        b_r_bar
          .attr("height", y2(jar3_r_chart.number + jar4_r_chart.number))
          .attr("y", y2(100 - (jar3_r_chart.number + jar4_r_chart.number)));

        b_r_label
          .text(jar3_r_chart.number + jar4_r_chart.number)
          .attr("y", y2(100 - (jar3_r_chart.number + jar4_r_chart.number)) - 5);

        b_g_bar
          .attr("height", y2(jar3_g_chart.number + jar4_g_chart.number))
          .attr("y", y2(100 - (jar3_g_chart.number + jar4_g_chart.number)));

        b_g_label
          .text(jar3_g_chart.number + jar4_g_chart.number)
          .attr("y", y2(100 - (jar3_g_chart.number + jar4_g_chart.number)) - 5);

        // Change the percentage labels
        var p_format = d3.format(".0%");
        var y3 = d3.scaleLinear()
            .rangeRound([jar_height - 5, 20])
            .domain([0, 1])
            .clamp(true);

        var jar1_p = jar1_r_chart.number / (jar1_r_chart.number + jar1_g_chart.number) || 0
        jar1_percent
          .text(p_format(jar1_p))
          .attr("y", y3(jar1_p));

        var jar2_p = jar2_r_chart.number / (jar2_r_chart.number + jar2_g_chart.number) || 0
        jar2_percent
          .text(p_format(jar2_p))
          .attr("y", y3(jar2_p));

        var jar3_p = jar3_r_chart.number / (jar3_r_chart.number + jar3_g_chart.number) || 0
        jar3_percent
          .text(p_format(jar3_p))
          .attr("y", y3(jar3_p));

        var jar4_p = jar4_r_chart.number / (jar4_r_chart.number + jar4_g_chart.number) || 0
        jar4_percent
          .text(p_format(jar4_p))
          .attr("y", y3(jar4_p));

        var y4 = d3.scaleLinear()
              .rangeRound([(jar_height * 2)-5, 20])
              .domain([0, 1]);

        var a_p = (jar1_r_chart.number + jar2_r_chart.number) / (jar1_r_chart.number + jar1_g_chart.number + jar2_r_chart.number + jar2_g_chart.number) || 0
        a_percent
          .text(p_format(a_p))
          .attr("y", y4(a_p));

        var b_p = (jar3_r_chart.number + jar4_r_chart.number) / (jar3_r_chart.number + jar3_g_chart.number + jar4_r_chart.number + jar4_g_chart.number) || 0
        b_percent
          .text(p_format(b_p))
          .attr("y", y4(b_p));

      };
    });
  };

  chart.bar_width = function(value) {
    bar_width = value;
    return chart;
  };


  chart.slider_position = function(value) {
    slider_position = value;
    return chart;
  };

  chart.bar_position = function(value) {
    bar_position = value;
    return chart;
  };

  chart.handle_position = function(value) {
    handle_position = value;
    return chart;
  };

  chart.bar_class = function(value) {
    bar_class = value;
    return chart;
  };

  chart.handle_class = function(value) {
    handle_class = value;
    return chart;
  };
 
  chart.bar_label_class = function(value) {
    bar_label_class = value;
    return chart;
  };

  chart.number = number; // Expose the number of jelly beans

  return chart;
};
