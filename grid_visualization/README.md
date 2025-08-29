Interactive 7×7 Grid Visualizer
================================

This small static page (see `index.html`) lets you experiment with patterns inspired by the 2025×2025 IMO grid tiling problem. It is designed to be hosted directly on GitHub Pages (no build step).

Features
--------
* 7×7 grid demo (adjustable in source by changing `const N = 7`).
* Click a cell to toggle it solid black (represents the uncovered square in that row/column).
* Click–drag from one cell to another to create an orange rectangular tile. Tiles cannot overlap black cells or existing tiles.
* Live status bar shows counts and whether each row/column currently has exactly one black cell.
* Remove a tile by clicking on it.
* Reset buttons to clear blacks, tiles, or everything.

Usage
-----
Open `index.html` locally in a browser or navigate to the corresponding GitHub Pages URL (e.g. `https://<username>.github.io/grid_visualization/`).

Controls
--------
* Single click: toggle black on a cell (blocked if the cell is under a tile).
* Drag (pointer down on a start cell, move, release): adds a tile if valid. Invalid (overlapping) rectangles show a red outline and are ignored on release.
* Click tile: remove tile.
* Keyboard: focus a cell (Tab) then press Space/Enter to toggle black.


Adapting to Larger Grids
------------------------
For exploration on larger boards, change `const N = 7` in `index.html`. Extremely large values (like 2025) will render but may be slow / large; for huge boards consider a canvas or virtualized approach.

Ideas / Next Steps
------------------
* Add a solver / helper to suggest tile placements.
* Provide a PNG / SVG export.
* Allow color-coding different tile sizes.
* Add a mode enforcing exactly one black per row/column automatically.

License
-------
Public domain / CC0. Use freely.

