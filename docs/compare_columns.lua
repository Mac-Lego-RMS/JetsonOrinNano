-- Shades the columns of comparison tables in the PDF: earlier vehicles in
-- grey, Napoleon in blue. A column is recognised by its header text, so a
-- table only has to name its columns consistently ("Regional final ...",
-- "National final ...", "Napoleon ...").

local FILL = {
  old = { head = 'luma(225)', body = 'luma(242)' },
  new = { head = 'rgb("#c9dbf5")', body = 'rgb("#e8f0fb")' },
}

local function kind(text)
  local t = text:lower()
  if t:find('napoleon') then return 'new' end
  if t:find('national final') or t:find('regional final') then return 'old' end
end

local function fill(cell, colour)
  cell.attr = pandoc.Attr(cell.attr.identifier, cell.attr.classes,
                          { ['typst:fill'] = colour })
end

function Table(tbl)
  local head = tbl.head.rows[1]
  if not head then return nil end
  local cols = {}
  for i, cell in ipairs(head.cells) do
    local k = kind(pandoc.utils.stringify(cell.contents))
    if k then
      cols[i] = k
      fill(cell, FILL[k].head)
    end
  end
  if next(cols) == nil then return nil end
  for _, body in ipairs(tbl.bodies) do
    for _, row in ipairs(body.body) do
      for i, cell in ipairs(row.cells) do
        if cols[i] then fill(cell, FILL[cols[i]].body) end
      end
    end
  end
  return tbl
end
