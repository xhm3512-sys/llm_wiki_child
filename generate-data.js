#!/usr/bin/env node
// 生成知识库搜索数据：扫描 concept / entity / source 目录及 index.md，
// 解析 wiki 链接 [[xxx]] 与 frontmatter，输出 search-data.js 供前端使用。

const fs = require('fs');
const path = require('path');

const ROOT = path.resolve(__dirname);
const INDEX = path.join(ROOT, 'index.md');
const CONCEPT_DIR = path.join(ROOT, 'concept');
const ENTITY_DIR = path.join(ROOT, 'entity');
const SOURCE_DIR = path.join(ROOT, 'source');

// 读取目录下所有 md 文件名
function listMd(dir) {
  if (!fs.existsSync(dir)) return [];
  return fs.readdirSync(dir).filter(f => f.endsWith('.md'));
}

// 去除文件名后缀
function stripMd(f) {
  return f.replace(/\.md$/, '');
}

// 提取所有 [[xxx]] 链接（去重保序）
function extractLinks(text) {
  const re = /\[\[([^\]]+)\]\]/g;
  const out = [];
  let m;
  while ((m = re.exec(text)) !== null) {
    const name = m[1].trim();
    if (name && !out.includes(name)) out.push(name);
  }
  return out;
}

// 解析 frontmatter（仅支持简单的 key: value 形式）
function parseFrontmatter(text) {
  const fm = {};
  const m = text.match(/^---\r?\n([\s\S]*?)\r?\n---/);
  if (!m) return { frontmatter: fm, body: text };
  const block = m[1];
  const body = text.slice(m[0].length).replace(/^\r?\n/, '');
  block.split(/\r?\n/).forEach(line => {
    const idx = line.indexOf(':');
    if (idx > -1) {
      const k = line.slice(0, idx).trim();
      const v = line.slice(idx + 1).trim();
      fm[k] = v;
    }
  });
  return { frontmatter: fm, body };
}

// 按二级/三级标题切分正文
function parseSections(body) {
  const sections = {};
  let current = null;
  let buf = [];
  const flush = () => {
    if (current) sections[current] = buf.join('\n').trim();
    buf = [];
  };
  body.split(/\r?\n/).forEach(line => {
    const h = line.match(/^(#{2,3})\s+(.*)$/);
    if (h) {
      flush();
      current = h[2].replace(/[#*]/g, '').trim();
    } else if (current) {
      buf.push(line);
    }
  });
  flush();
  return sections;
}

// 从 source 文件解析结构化字段
function parseSourceFile(file, dir) {
  const full = path.join(dir, file);
  const raw = fs.readFileSync(full, 'utf8');
  const { body } = parseFrontmatter(raw);
  const sections = parseSections(body);
  const title = stripMd(file);

  // 互动数据：从"基本信息"中提取点赞、收藏、评论
  const baseInfo = sections['基本信息'] || '';
  const stats = {};
  baseInfo.split(/\r?\n/).forEach(line => {
    const m = line.match(/\*\*(\w+)\*\*[：:]?\s*([^、,，]+)/);
    if (m) stats[m[1]] = m[2].trim();
  });

  // 相关实体 / 关联概念：按行提取 [[xxx]]
  const relatedEntities = extractLinks(sections['相关实体'] || '');
  const relatedConcepts = extractLinks(sections['关联概念'] || '');
  const keywords = extractLinks(sections['关键词标签'] || '');

  return {
    id: 'source/' + title,
    type: 'source',
    title,
    file: 'source/' + file,
    summary: (sections['核心主题'] || '').replace(/\[\[|\]\]/g, ''),
    keyPoints: (sections['关键要点'] || '').replace(/\[\[|\]\]/g, ''),
    keywords,
    relatedEntities,
    relatedConcepts,
    stats,
    links: extractLinks(body),
    body
  };
}

// 解析 concept / entity 文件
function parseWikiFile(file, dir, type) {
  const full = path.join(dir, file);
  const raw = fs.readFileSync(full, 'utf8');
  const { frontmatter, body } = parseFrontmatter(raw);
  const title = stripMd(file);
  return {
    id: type + '/' + title,
    type,
    title,
    file: type + '/' + file,
    aliases: [],
    description: body.replace(/\[\[|\]\]/g, '').trim(),
    links: extractLinks(body),
    body,
    frontmatter
  };
}

// 解析 index.md 中的别名表与分类目录
function parseIndex() {
  const raw = fs.readFileSync(INDEX, 'utf8');
  const aliases = {};
  // 别名表：| 别名 | 指向 |
  const re = /^\|([^|]+)\|\s*\[\[([^\]]+)\]\]\s*\|/gm;
  let m;
  while ((m = re.exec(raw)) !== null) {
    const alias = m[1].trim();
    const target = m[2].trim();
    if (alias && target) aliases[alias.toLowerCase()] = target;
  }

  // 分类目录：从各类别的小节中提取条目
  // 行格式：- [[concept/xxx]] — 描述  或  - [[entity/xxx]] — 描述
  const entries = [];
  const lines = raw.split(/\r?\n/);
  let currentCategory = '';
  let currentType = '';
  for (const line of lines) {
    const cat = line.match(/^###\s+(\d+\.\s*.+)$/);
    if (cat) {
      currentCategory = cat[1].replace(/^\d+\.\s*/, '').trim();
      continue;
    }
    const major = line.match(/^##\s+([一二三四五]、(.+))/);
    if (major) {
      // 用一级标题切换 type
      if (/概念/.test(major[2])) currentType = 'concept';
      else if (/实体/.test(major[2])) currentType = 'entity';
      else if (/原始资料/.test(major[2])) currentType = 'source';
      else currentType = '';
      currentCategory = '';
      continue;
    }
    const item = line.match(/^-\s*\[\[([^\]]+)\]\]\s*[—\-]?\s*(.*)$/);
    if (item && currentType) {
      entries.push({
        link: item[1],
        type: currentType,
        category: currentCategory,
        desc: item[2].trim()
      });
    }
  }
  return { aliases, entries };
}

function main() {
  const output = {
    generatedAt: new Date().toISOString(),
    concepts: [],
    entities: [],
    sources: [],
    aliases: {},
    indexEntries: []
  };

  // concept
  listMd(CONCEPT_DIR).forEach(f => {
    output.concepts.push(parseWikiFile(f, CONCEPT_DIR, 'concept'));
  });
  // entity
  listMd(ENTITY_DIR).forEach(f => {
    output.entities.push(parseWikiFile(f, ENTITY_DIR, 'entity'));
  });
  // source
  listMd(SOURCE_DIR).forEach(f => {
    output.sources.push(parseSourceFile(f, SOURCE_DIR));
  });

  const idx = parseIndex();
  output.aliases = idx.aliases;
  output.indexEntries = idx.entries;

  // 把 index 中条目的 desc / category 回填到对应 wiki 条目
  const byTitle = {};
  [...output.concepts, ...output.entities].forEach(c => { byTitle[c.title] = c; });
  idx.entries.forEach(e => {
    const name = e.link.split('/').pop();
    if (byTitle[name]) {
      byTitle[name].description = byTitle[name].description || e.desc;
      byTitle[name].category = e.category;
    }
  });

  // 统计每个条目被引用次数（反向链接强度）
  const refCount = {};
  const all = [...output.concepts, ...output.entities, ...output.sources];
  all.forEach(item => {
    (item.links || []).forEach(l => {
      refCount[l] = (refCount[l] || 0) + 1;
    });
  });
  [...output.concepts, ...output.entities].forEach(c => {
    c.refCount = refCount[c.title] || 0;
  });

  // ===== 为支持向量/语义搜索，预计算 TF-IDF =====
  // 1) 中文分词：采用 bigram（连续两字）+ 英文单词 的混合策略，
  //    对知识库这种小规模、无字典场景效果稳定。
  // 2) 为每个条目计算 TF-IDF 稀疏向量，输出词表 + IDF 供前端查询时复用。
  const STOP_CHARS = /\s|\n|\r|\t|[，。、；：！？“”‘’（）【】《》「」\[\]|·\-—…~`!@#$%^&*()_+=\-{}\|;:'",.<>/?\\0-9]/;
  function tokenize(text) {
    if (!text) return [];
    const tokens = [];
    // 先按非中英文字符切分，再对每段做处理
    const segs = text.split(STOP_CHARS).filter(Boolean);
    segs.forEach(seg => {
      // 英文/拉丁部分：按小写单词
      if (/^[a-zA-Z]+$/.test(seg)) {
        tokens.push(seg.toLowerCase());
        return;
      }
      // 中文部分：bigram 切分
      const chars = Array.from(seg);
      for (let i = 0; i < chars.length - 1; i++) {
        const bigram = chars[i] + chars[i + 1];
        // 过滤含标点的 bigram
        if (!STOP_CHARS.test(chars[i]) && !STOP_CHARS.test(chars[i + 1])) {
          tokens.push(bigram);
        }
      }
      // 单字也保留（针对短词，如"门将"会被切成 bigram"门将"，但"门"单字也加入提升召回）
      chars.forEach(c => { if (!STOP_CHARS.test(c)) tokens.push(c); });
    });
    return tokens;
  }

  // 为每个条目拼接用于建索引的全文文本
  function docText(item) {
    const parts = [item.title];
    if (item.description) parts.push(item.description);
    if (item.summary) parts.push(item.summary);
    if (item.keyPoints) parts.push(item.keyPoints);
    if (item.keywords && item.keywords.length) parts.push(item.keywords.join(' '));
    if (item.relatedEntities && item.relatedEntities.length) parts.push(item.relatedEntities.join(' '));
    if (item.relatedConcepts && item.relatedConcepts.length) parts.push(item.relatedConcepts.join(' '));
    if (item.links && item.links.length) parts.push(item.links.join(' '));
    if (item.category) parts.push(item.category);
    if (item.body) parts.push(item.body.replace(/\[\[|\]\]/g, ''));
    return parts.join(' ');
  }

  // 统计文档频率
  const docFreq = {};
  const docTokens = all.map(item => {
    const tokens = tokenize(docText(item));
    // 词频
    const tf = {};
    tokens.forEach(t => { tf[t] = (tf[t] || 0) + 1; });
    Object.keys(tf).forEach(t => { docFreq[t] = (docFreq[t] || 0) + 1; });
    item._tf = tf;
    item._tokenCount = tokens.length;
    return item;
  });

  const N = docTokens.length;
  // 构建词表（按文档频率过滤：出现 >=2 次的词才进入词表，避免噪声）
  const vocab = {};
  let vocabIdx = 0;
  Object.keys(docFreq).forEach(t => {
    if (docFreq[t] >= 2) {
      vocab[t] = vocabIdx++;
    }
  });
  // IDF: log((N+1)/(df+1)) + 1 （平滑）
  const idf = {};
  Object.keys(vocab).forEach(t => {
    idf[t] = Math.log((N + 1) / (docFreq[t] + 1)) + 1;
  });

  // 为每个条目生成稀疏 TF-IDF 向量 { tokenIndex: weight }
  // 同时保存归一化向量，便于前端做余弦相似度
  docTokens.forEach(item => {
    const vec = {};
    const tf = item._tf;
    let norm = 0;
    Object.keys(tf).forEach(t => {
      if (vocab[t] === undefined) return;
      const weight = tf[t] * idf[t];
      vec[vocab[t]] = weight;
      norm += weight * weight;
    });
    norm = Math.sqrt(norm) || 1;
    // 归一化
    Object.keys(vec).forEach(k => { vec[k] = vec[k] / norm; });
    item.tfidf = vec;
    item.vecNorm = norm;
    // 清理临时字段
    delete item._tf;
    delete item._tokenCount;
  });

  // 把 vocab 保存为数组（index -> token），便于前端用稀疏向量查回词）
  const vocabArr = Object.keys(vocab).sort((a, b) => vocab[a] - vocab[b]);
  output.vocab = vocabArr;
  output.idf = vocabArr.map(t => idf[t]);

  const js = `// 自动生成，请勿手动编辑。由 generate-data.js 产出。
window.WIKI_DATA = ${JSON.stringify(output, null, 0)};
`;
  const outPath = path.join(ROOT, 'search-data.js');
  fs.writeFileSync(outPath, js, 'utf8');

  console.log('生成完成：' + outPath);
  console.log('  concepts: ' + output.concepts.length);
  console.log('  entities: ' + output.entities.length);
  console.log('  sources:  ' + output.sources.length);
  console.log('  aliases:  ' + Object.keys(output.aliases).length);
  console.log('  indexEntries: ' + output.indexEntries.length);
  console.log('  vocab(向量词表): ' + output.vocab.length);
}

main();
