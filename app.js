window.parseInput = function parseInput(text) {
  const lines = String(text || '')
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);

  return {
    count: lines.length,
    instruments: lines.map(parseInstrumentLine)
  };
};

function parseInstrumentLine(line) {
  const tuningMatch = line.match(/\bin\s+([A-G](?:b|#)?)/i);
  const numberMatch = line.match(/\b(\d+)\b/);

  return {
    raw: line,
    name: normalizeInstrumentName(line),
    number: numberMatch ? Number(numberMatch[1]) : null,
    transposition: tuningMatch ? tuningMatch[1] : null
  };
}

function normalizeInstrumentName(line) {
  return String(line)
    .replace(/\bin\s+[A-G](?:b|#)?\b/gi, '')
    .replace(/\b\d+\b/g, '')
    .replace(/\s+/g, ' ')
    .trim();
}
