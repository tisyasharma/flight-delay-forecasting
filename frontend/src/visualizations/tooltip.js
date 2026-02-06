// shared tooltip for D3 charts, lazily creates the DOM element on first use

import * as d3 from 'd3';

let tooltip = null;

function getTooltip() {
  if (!tooltip) {
    tooltip = d3
      .select('body')
      .append('div')
      .attr('class', 'tooltip')
      .style('opacity', 0)
      .style('pointer-events', 'none');
  }
  return tooltip;
}

// show tooltip near cursor, clamped to optional bounds element
export function showTooltip(event, content, options = {}) {
  const tip = getTooltip();
  const offsetX = Number.isFinite(options.offsetX) ? options.offsetX : 16;
  const offsetY = Number.isFinite(options.offsetY) ? options.offsetY : 16;
  const padding = 8;
  const boundsRect = options.bounds?.getBoundingClientRect
    ? options.bounds.getBoundingClientRect()
    : { left: 0, right: window.innerWidth, top: 0, bottom: window.innerHeight };
  const clientX = Number.isFinite(event?.clientX) ? event.clientX : event.pageX - window.scrollX;
  const clientY = Number.isFinite(event?.clientY) ? event.clientY : event.pageY - window.scrollY;
  tip
    .html(content)
    .style('opacity', 1)
    .style('display', 'block');
  const rect = tip.node().getBoundingClientRect();
  let x = clientX + offsetX;
  let y = clientY + offsetY;
  if (x + rect.width + padding > boundsRect.right) {
    x = clientX - offsetX - rect.width;
  }
  if (x < boundsRect.left + padding) {
    x = boundsRect.left + padding;
  }
  if (y + rect.height + padding > boundsRect.bottom) {
    y = clientY - offsetY - rect.height;
  }
  if (y < boundsRect.top + padding) {
    y = boundsRect.top + padding;
  }
  tip
    .style('left', `${x + window.scrollX}px`)
    .style('top', `${y + window.scrollY}px`);
}

export function hideTooltip() {
  const tip = getTooltip();
  tip.style('opacity', 0).style('display', 'none');
}
