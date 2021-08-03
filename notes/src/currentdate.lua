-- Put the current date in place of the text {{currentdate}} anywhere in the document.
-- Adapted from here: https://pandoc.org/lua-filters.html#replacing-placeholders-with-their-metadata-value
-- https://pandoc.org/lua-filters.html
-- When using pandoc call like this pandoc --lua-filter=currentdate.lua
return {
  {
    Str = function (elem)
      if elem.text == "{{currentdate}}" then
        return pandoc.Str(os.date("%m/%d/%Y %I:%M %p"))
      else
        return elem
      end
    end,
  }
}