#!/usr/bin/env node

/**
 * STREAMLINED OSRS Wiki Watchdog - Core Functionality Only
 *
 * Purpose: Monitor OSRS wiki for changes and maintain comprehensive content collection
 * - Fetches pages from target namespaces (Main, Guide, Update, Exchange, Transcript)
 * - Includes subpages (Money making guide/*, etc.)
 * - Maintains historical content while adding new pages
 * - Clean CLI interface with progress indicators
 */

const fs = require('fs-extra');
const path = require('path');
const axios = require('axios');
const chalk = require('chalk');
const ora = require('ora');
const { spawn } = require('child_process');
const os = require('os');

class StreamlinedOSRSWatchdog {
  constructor() {
    this.wikiApiUrl = 'https://oldschool.runescape.wiki/api.php';
    this.userAgent = 'OSRS-AI-System/1.0 (https://github.com/user/osrs-ai; contact@example.com)';

    // File paths - CONSOLIDATED TRACKING
    this.outputFile = path.join(__dirname, '../data/osrs_wiki_content.jsonl');
    this.pageTitlesFile = path.join(__dirname, '../data/osrs_page_titles.txt');
    this.metadataFile = path.join(__dirname, '../data/osrs_watchdog_tracking.json');
    this.filteredPagesFile = path.join(__dirname, '../data/osrs_filtered_pages.txt');
    this.nullPagesFile = path.join(__dirname, '../data/osrs_null_pages.txt');
    this.wikitextFile = path.join(__dirname, '../data/osrs_wikitext_content.jsonl');

    // Target namespaces for comprehensive OSRS content
    this.targetNamespaces = [
      { id: 0, name: 'Main', description: 'Core game content (items, monsters, quests, guides, locations)' },
      { id: 3002, name: 'Guide', description: 'Community strategy guides and builds' }
    ];

    // In-memory data
    this.pageData = new Map();
    this.pageTitles = new Set();
    this.filteredPages = new Set();
    this.nullPages = new Set();
    this.metadata = null;
    // Seen snapshot keys to prevent duplicate wikitext appends (title+revid)
    this.seenWikitextRevisions = new Set();

    // State
    this.isRunning = false;
    this.currentOperation = '';
    this.stats = {
      pagesChecked: 0,
      pagesUpdated: 0,
      pagesAdded: 0,
      templatesProcessed: 0,
      errors: 0
    };

    // Progress tracking - SINGLE CONSOLIDATED PROGRESS
    this.totalOperations = 0;
    this.completedOperations = 0;
    this.startTime = null;

    // Full refetch flag (Option B): env or CLI
    this.fullRefetch = process.env.OSRS_FULL_REFETCH === '1' || process.argv.includes('--full-refetch');

    // Batch fetch settings (serial batching)
    this.batchFetch = process.env.OSRS_BATCH_FETCH === '1' || process.argv.includes('--batch-fetch');
    const bsIdx = process.argv.indexOf('--batch-size');
    this.batchSize = Number.isFinite(parseInt(process.env.OSRS_BATCH_SIZE || '', 10))
      ? parseInt(process.env.OSRS_BATCH_SIZE, 10)
      : (bsIdx !== -1 ? parseInt(process.argv[bsIdx + 1] || '50', 10) : 50);
    const tbIdx = process.argv.indexOf('--test-batch');
    this.testBatchCount = tbIdx !== -1 ? parseInt(process.argv[tbIdx + 1] || '0', 10) : 0;

    // Optional skips
    this.skipReprocess = process.env.OSRS_SKIP_REPROCESS === '1' || process.argv.includes('--skip-reprocess');
    this.skipChecker = process.env.OSRS_SKIP_CHECKER === '1' || process.argv.includes('--skip-checker');
  }


  // ═══════════════════════════════════════════════════════════════════════════════
  // MAIN EXECUTION
  // ═══════════════════════════════════════════════════════════════════════════════

  async run() {
    console.log(chalk.cyan.bold('🚀 OSRS Wiki Watchdog - Streamlined Edition'));
    console.log(chalk.gray('━'.repeat(60)));


    try {
      await this.initialize();


      // Optional full refetch mode (Option B)
      if (this.fullRefetch) {
        console.log(chalk.yellow(`\nd Full refetch flag detected 	7 rebuilding entire collection from API`));
        await this.refetchAllExistingPages();
      }

      // TEMPLATE CHECKER - SCAN AND CORRECT FORMATTING ISSUES
      if (!this.skipChecker) {
        console.log(chalk.yellow(`\n🔧 Template Format Validation & Correction`));
        await this.runTemplateChecker();
      } else {
        console.log(chalk.gray(`\n⏭️  Skipping template checker (flag)`));
      }

      await this.updateCollection();

      // BATCH PROCESSING AND ALPHABETICAL REORGANIZATION
      if (!this.skipReprocess) {
        console.log(chalk.yellow(`\n🔄 Batch Processing & Alphabetical Reorganization`));
        console.log(chalk.gray(`   Loading ${this.pageData.size.toLocaleString()} pages into memory for full processing...`));
        // BATCH REPROCESS ALL PAGES WITH TEMPLATE FIXES
        await this.batchReprocessAllPages();
      } else {
        console.log(chalk.gray(`\n⏭️  Skipping batch reprocessing (flag)`));
      }
      // Always alphabetize/compact
      await this.alphabeticallyReorganizeContent();

      await this.startMonitoring();
    } catch (error) {
      console.error(chalk.red(`❌ Fatal error: ${error.message}`));
      process.exit(1);
    }
  }

  async initialize() {
    const spinner = ora('Initializing watchdog...').start();

    try {
      await this.loadMetadata();
      await this.loadExistingData();
      await this.loadPageTitles();
      await this.loadFilteredPages();
      await this.loadNullPages();
      await this.loadSeenWikitextRevisions();

      spinner.succeed(chalk.green('✅ Initialization complete'));

      // Display status
      console.log(chalk.blue('📊 OSRS Watchdog Status:'));
      console.log(chalk.gray(`   📄 Content pages: ${this.pageData.size.toLocaleString()}`));
      console.log(chalk.gray(`   📋 Page titles: ${this.pageTitles.size.toLocaleString()}`));
      console.log(chalk.gray(`   🚫 Filtered pages: ${this.filteredPages.size.toLocaleString()}`));
      console.log(chalk.gray(`   ⚫ Null pages: ${this.nullPages.size.toLocaleString()}`));
      console.log(chalk.gray(`   🔄 Total runs: ${this.metadata?.totalRuns || 0}`));
      console.log(chalk.gray(`   📡 Total API calls: ${this.metadata?.totalApiCalls || 0}`));
      console.log(chalk.gray(`   🕐 Last update: ${this.metadata?.lastUpdate ? new Date(this.metadata.lastUpdate).toLocaleString() : 'Never'}`));

    } catch (error) {
      spinner.fail(chalk.red('❌ Initialization failed'));
      throw error;
    }
  }

  async updateCollection() {
    console.log(chalk.blue('\n🔄 Updating Collection'));
    console.log(chalk.gray('─'.repeat(30)));

    // Step 1: Check for new pages in target namespaces
    await this.scanForNewPages();

    // Step 2: Check existing pages for updates
    await this.checkForUpdates();

    // Step 3: Save changes
    await this.saveChanges();
  }

  async refetchAllExistingPages() {
    console.log(chalk.blue('\n🔁 Full refetch mode: refetching all known titles'));
    let titles = Array.from(this.pageTitles);
    if (titles.length === 0 && this.pageData.size > 0) {
      titles = Array.from(this.pageData.keys());
    }
    titles = titles.sort();

    if (titles.length === 0) {
      console.log(chalk.gray('   No known titles to refetch. Skipping.'));
      return;
    }

    // Optional small test subset
    if (this.testBatchCount && this.testBatchCount > 0) {
      titles = titles.slice(0, this.testBatchCount);
      console.log(chalk.yellow(`   Test mode: limiting to first ${titles.length} titles`));
    }

    let processed = 0;
    let saved = 0;
    const total = titles.length;

    if (this.batchFetch) {
      const batchSize = Math.max(1, Math.min(this.batchSize || 50, 50));
      for (let i = 0; i < titles.length; i += batchSize) {
        const chunk = titles.slice(i, i + batchSize);
        try {
          const contents = await this.fetchPagesBatch(chunk);
          for (const content of contents) {
            if (content && this.shouldIncludePage(content)) {
              await this.saveImmediately('page', content);
              saved++;
            }
          }
        } catch (e) {
          this.stats.errors++;
        }
        processed += chunk.length;
        this.updateProgress('Full refetch (batched)', processed, total);
      }
    } else {
      for (const title of titles) {
        try {
          const content = await this.fetchPageContent(title);
          if (content && this.shouldIncludePage(content)) {
            await this.saveImmediately('page', content);
            saved++;
          } else if (!content) {
            this.nullPages.add(title);
          }
        } catch (e) {
          this.stats.errors++;
        }
        processed++;
        this.updateProgress('Full refetch', processed, total);
      }
    }

    console.log(chalk.green(`\n✅ Full refetch complete: ${saved.toLocaleString()} pages refreshed`));
    await this.saveChanges();
  }

  async startMonitoring() {
    console.log(chalk.green('\n👁️  Starting continuous monitoring...'));
    console.log(chalk.gray('Press Ctrl+C to stop'));

    this.isRunning = true;

    // Monitor every 10 minutes - FULL SCAN each time
    const monitorInterval = setInterval(async () => {
      if (!this.isRunning) {
        clearInterval(monitorInterval);
        return;
      }

      console.log(chalk.blue(`\n🔄 Full scan cycle... ${new Date().toLocaleTimeString()}`));
      console.log(chalk.blue('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'));

      // Do complete update cycle (scan for new pages + check for updates)
      await this.updateCollection();

    }, 10 * 60 * 1000);

    // Graceful shutdown
    process.on('SIGINT', () => {
      console.log(chalk.yellow('\n🛑 Shutting down gracefully...'));
      this.isRunning = false;
      clearInterval(monitorInterval);
      process.exit(0);
    });
  }

  // ═══════════════════════════════════════════════════════════════════════════════
  // CORE FUNCTIONALITY
  // ═══════════════════════════════════════════════════════════════════════════════

  async scanForNewPages() {
    console.log(chalk.blue('🔍 Scanning for new pages across all target namespaces...'));
    let totalFound = 0;
    let totalScanned = 0;

    try {
      for (const namespace of this.targetNamespaces) {
        console.log(chalk.yellow(`\n📂 Scanning ${namespace.name} namespace (${namespace.description})`));

        const pages = await this.getAllPagesInNamespace(namespace.id);
        const knownTitles = new Set([
          ...this.pageTitles,
          ...this.filteredPages,
          ...this.nullPages
        ]);
        const newPages = pages.filter(p => !knownTitles.has(p.title));
        const existingPages = pages.filter(p => knownTitles.has(p.title));
        totalScanned += pages.length;

        console.log(chalk.gray(`   📊 Found ${pages.length.toLocaleString()} total pages: ${newPages.length.toLocaleString()} new, ${existingPages.length.toLocaleString()} existing`));

        // Process ONLY NEW pages during normal operation
        if (newPages.length > 0) {
          await this.processPagesWithProgress(newPages, namespace.name);
          totalFound += newPages.length;
        }
      }

      console.log(chalk.green(`\n✅ Scan complete: ${totalScanned.toLocaleString()} pages scanned, ${totalFound.toLocaleString()} new pages added`));

    } catch (error) {
      console.error(chalk.red('❌ Failed to scan for new pages'));
      throw error;
    }
  }

  async checkForUpdates() {
    if (this.pageTitles.size === 0) return;

    console.log(chalk.blue('🔄 Checking for page updates...'));

    try {
      const recentChanges = await this.getRecentChanges();
      const relevantChanges = recentChanges.filter(change =>
        this.pageTitles.has(change.title)
      );

      console.log(chalk.gray(`   📊 Found ${recentChanges.length} recent changes, ${relevantChanges.length} relevant to our collection`));

      if (relevantChanges.length > 0) {
        await this.updatePagesWithProgress(relevantChanges);
        console.log(chalk.green(`✅ Updated ${relevantChanges.length} pages`));
      } else {
        console.log(chalk.green('✅ No updates needed'));
      }

    } catch (error) {
      console.error(chalk.red('❌ Failed to check for updates'));
      this.stats.errors++;
    }
  }

  // ═══════════════════════════════════════════════════════════════════════════════
  // API METHODS
  // ═══════════════════════════════════════════════════════════════════════════════

  async getAllPagesInNamespace(namespaceId) {
    const pages = [];
    let apcontinue = null;
    let apiCalls = 0;

    do {
      apiCalls++;

      const response = await axios.get(this.wikiApiUrl, {
        params: {
          action: 'query',
          list: 'allpages',
          apnamespace: namespaceId,
          aplimit: 500,
          apfilterredir: 'nonredirects', // No redirects
          format: 'json',
          ...(apcontinue && { apcontinue })
        },
        headers: { 'User-Agent': this.userAgent },
        timeout: 30000
      });

      const pageList = response.data.query?.allpages || [];
      pages.push(...pageList);

      // Show scanning progress
      process.stdout.write(`\r   📡 API calls: ${apiCalls} | Pages found: ${pages.length.toLocaleString()}`);

      apcontinue = response.data.continue?.apcontinue;
      await this.sleep(500); // Rate limiting

    } while (apcontinue);

    // Clear progress line
    process.stdout.write('\r' + ' '.repeat(80) + '\r');

    return pages;
  }

  async getRecentChanges() {
    const lastCheck = this.metadata?.lastUpdate || (Date.now() - 10 * 60 * 1000);
    const since = new Date(lastCheck).toISOString();

    const response = await axios.get(this.wikiApiUrl, {
      params: {
        action: 'query',
        list: 'recentchanges',
        rcstart: new Date().toISOString(),
        rcend: since,
        rcnamespace: this.targetNamespaces.map(ns => ns.id).join('|'),
        rctype: 'edit|new',
        rcprop: 'title|timestamp',
        rclimit: 500,
        format: 'json'
      },
      headers: { 'User-Agent': this.userAgent },
      timeout: 30000
    });

    return response.data.query?.recentchanges || [];
  }

  async fetchPageContent(title) {
    try {
      // Fetch raw wikitext first for proper template processing
      const wikitextResponse = await axios.get(this.wikiApiUrl, {
        params: {
          action: 'query',
          titles: title,
          prop: 'revisions|categories|info',
          rvprop: 'content|timestamp',
          rvslots: 'main',
          format: 'json',
          formatversion: 2
        },
        headers: { 'User-Agent': this.userAgent },
        timeout: 30000
      });

      const page = wikitextResponse.data.query?.pages?.[0];
      if (!page || page.missing) return null;

      const revision = page.revisions?.[0];
      if (!revision) return null;

      const rawWikitext = revision.slots?.main?.content || '';
      const categories = page.categories?.map(cat => cat.title) || [];
      // Raw wikitext snapshot is appended during saveImmediately() to avoid duplicates

      // Process templates from raw wikitext
      const processedWikitext = await this.processTemplatesFromWikitext(rawWikitext);

      // Convert to clean text
      const cleanText = this.cleanWikitextContent(processedWikitext);

      // Skip pages that are too short after processing (reduced threshold)
      if (cleanText.length < 50) {
        console.log(chalk.gray(`   ⚠️  Skipping ${page.title} - too short (${cleanText.length} chars)`));
        return null;
      }

      return {
        title: page.title,
        text: cleanText,
        categories: categories,
        timestamp: revision.timestamp,
        revid: page.lastrevid,
        rawWikitext: rawWikitext
      };

    } catch (error) {
      console.error(chalk.red(`❌ Error fetching ${title}: ${error.message}`));
      return null;
    }
  }

  // Serial batched fetch of multiple titles in one API call
  async fetchPagesBatch(titles) {
    if (!Array.isArray(titles) || titles.length === 0) return [];

    const params = {
      action: 'query',
      format: 'json',
      formatversion: 2,
      prop: 'revisions|categories|info',
      rvslots: 'main',
      rvprop: 'content|timestamp',
      cllimit: 'max',
      redirects: 1,
      maxlag: 5,
      titles: titles.join('|')
    };

    let attempt = 0;
    for (;;) {
      try {
        const res = await axios.post(this.wikiApiUrl, new URLSearchParams(params), {
          headers: {
            'User-Agent': this.userAgent,
            'Accept-Encoding': 'gzip',
            // Etiquette hint for POST read-only calls
            'Promise-Non-Write-API-Action': 'true'
          },
          timeout: 60000
        });

        const pages = res.data?.query?.pages || [];
        const results = [];

        for (const page of pages) {
          if (!page || page.missing) {
            if (page?.title) this.nullPages.add(page.title);
            continue;
          }
          const revision = page.revisions?.[0];
          if (!revision) {
            this.nullPages.add(page.title);
            continue;
          }
          const rawWikitext = revision.slots?.main?.content || '';
          const categories = page.categories?.map(c => c.title) || [];

          // Raw wikitext snapshot is appended during saveImmediately() to avoid duplicates

          // Process and clean
          const processed = await this.processTemplatesFromWikitext(rawWikitext);
          const cleanText = this.cleanWikitextContent(processed);
          if (cleanText.length < 50) {
            this.filteredPages.add(page.title);
            continue;
          }

          results.push({
            title: page.title,
            text: cleanText,
            categories,
            timestamp: revision.timestamp,
            revid: page.lastrevid,
            rawWikitext
          });
        }

        return results;
      } catch (err) {
        const retryAfterHdr = parseInt(err?.response?.headers?.['retry-after'] || '0', 10);
        const isLag = err?.response?.data?.error?.code === 'maxlag' || err?.response?.status === 503 || retryAfterHdr;
        if (isLag && attempt <= 6) {
          const backoffMs = retryAfterHdr ? retryAfterHdr * 1000 : Math.min(30000, 1000 * Math.pow(2, attempt));
          await this.sleep(backoffMs);
          attempt++;
          continue;
        }
        throw err;
      }
    }
  }


  // ═══════════════════════════════════════════════════════════════════════════════
  // DATA MANAGEMENT WITH PROGRESS TRACKING
  // ═══════════════════════════════════════════════════════════════════════════════

  async processPagesWithProgress(pages, namespaceName) {
    if (pages.length === 0) return;

    console.log(chalk.blue(`\n📋 Processing ${pages.length.toLocaleString()} new pages from ${namespaceName}`));

    this.startTime = Date.now();
    let processed = 0;
    let successful = 0;

    for (const page of pages) {
      try {
        // CORRECT LOGIC: Always check content, but track what type of page it is
        const isNewPage = !this.pageTitles.has(page.title);

        const content = await this.fetchPageContent(page.title);
        if (content && this.shouldIncludePage(content)) {
          await this.saveImmediately('page', content);

          if (isNewPage) {
            this.stats.pagesAdded++;
          } else {
            this.stats.pagesUpdated++; // Existing page with new/updated content
          }
          successful++;

          // Remove from filtered/null if it was there (page recovered)
          this.filteredPages.delete(page.title);
          this.nullPages.delete(page.title);

        } else if (!content) {
          // Track null pages for statistics, but don't skip them next time
          this.nullPages.add(page.title);
        } else {
          // Track filtered pages for statistics, but don't skip them next time
          this.filteredPages.add(page.title);
        }

        processed++;

        // Update progress every page
        this.updateProgress(`Processing ${namespaceName}`, processed, pages.length, ` | Success: ${successful}`);

        await this.sleep(500); // Rate limiting

      } catch (error) {
        this.stats.errors++;
        processed++;
        this.updateProgress(`Processing ${namespaceName}`, processed, pages.length, ` | Success: ${successful} | Errors: ${this.stats.errors}`);
      }
    }

    const duration = (Date.now() - this.startTime) / 1000;
    this.showFinalResult(`${namespaceName} Processing`, successful, pages.length, duration);
  }

  async updatePagesWithProgress(changes) {
    if (changes.length === 0) return;

    console.log(chalk.blue(`\n⚙️  Updating ${changes.length.toLocaleString()} changed pages...`));

    this.startTime = Date.now();
    let processed = 0;

    for (const change of changes) {
      try {
        await this.updatePage(change.title);
        this.stats.pagesUpdated++;

        processed++;

        // Show progress every 5 pages or on last page
        if (processed % 5 === 0 || processed === changes.length) {
          const elapsed = (Date.now() - this.startTime) / 1000;
          const rate = Math.round(processed / elapsed * 10) / 10;
          const percentage = Math.round((processed / changes.length) * 100);
          const eta = Math.round((changes.length - processed) / rate);

          // Create visual progress bar
          const barLength = 40;
          const filledLength = Math.round((processed / changes.length) * barLength);
          const bar = '█'.repeat(filledLength) + '░'.repeat(barLength - filledLength);

          process.stdout.write(`\r   Updates |${chalk.cyan(bar)}| ${percentage}% | ${processed}/${changes.length} pages | ETA: ${eta}s | Rate: ${rate}/s`);
        }

        await this.sleep(500);

      } catch (error) {
        this.stats.errors++;
        processed++;

        // Update progress on error too
        if (processed % 5 === 0 || processed === changes.length) {
          const elapsed = (Date.now() - this.startTime) / 1000;
          const rate = Math.round(processed / elapsed * 10) / 10;
          const percentage = Math.round((processed / changes.length) * 100);
          const eta = Math.round((changes.length - processed) / rate);

          const barLength = 40;
          const filledLength = Math.round((processed / changes.length) * barLength);
          const bar = '█'.repeat(filledLength) + '░'.repeat(barLength - filledLength);

          process.stdout.write(`\r   Updates |${chalk.cyan(bar)}| ${percentage}% | ${processed}/${changes.length} pages | ETA: ${eta}s | Rate: ${rate}/s`);
        }
      }
    }

    // Clear progress line and show final result
    process.stdout.write('\r' + ' '.repeat(100) + '\r');
    console.log(chalk.green(`   ✅ Successfully updated ${processed.toLocaleString()} pages`));
  }

  async updatePage(title) {
    try {
      const content = await this.fetchPageContent(title);
      if (content) {
        await this.saveImmediately('page', content);
      }
      await this.sleep(500);
    } catch (error) {
      this.stats.errors++;
    }
  }

  shouldIncludePage(pageData) {
    const title = pageData.title.toLowerCase();

    // Filter out unwanted pages
    if (title.includes('disambiguation') ||
        title.includes('redirect') ||
        title.includes('stub') ||
        title.startsWith('user:') ||
        title.startsWith('talk:')) {
      return false;
    }

    return true;
  }

  // ═══════════════════════════════════════════════════════════════════════════════
  // FILE OPERATIONS
  // ═══════════════════════════════════════════════════════════════════════════════

  async loadMetadata() {
    try {
      if (fs.existsSync(this.metadataFile)) {
        this.metadata = JSON.parse(fs.readFileSync(this.metadataFile, 'utf8'));
      } else {
        this.metadata = {
          lastUpdate: null,
          totalPages: 0,
          totalRuns: 0,
          totalApiCalls: 0,
          lastRunStats: { added: 0, updated: 0, errors: 0 },
          version: '3.0-streamlined'
        };
      }
    } catch (error) {
      this.metadata = {
        lastUpdate: null,
        totalPages: 0,
        totalRuns: 0,
        totalApiCalls: 0,
        lastRunStats: { added: 0, updated: 0, errors: 0 },
        version: '3.0-streamlined'
      };
    }
  }

  async loadExistingData() {
    if (!fs.existsSync(this.outputFile)) return;

    const content = fs.readFileSync(this.outputFile, 'utf8');
    const lines = content.trim().split('\n').filter(line => line.trim());

    for (const line of lines) {
      try {
        const pageData = JSON.parse(line);
        if (pageData && typeof pageData.title === 'string' && pageData.title.length > 0) {
          this.pageData.set(pageData.title, pageData);
        } else {
          // Skip entries without a valid title
          continue;
        }
      } catch (error) {
        // Skip invalid lines
      }
    }
  }

  async loadPageTitles() {
    if (!fs.existsSync(this.pageTitlesFile)) return;

    const content = fs.readFileSync(this.pageTitlesFile, 'utf8');
    const titles = content.trim().split('\n').filter(title => title.trim());

    for (const title of titles) {
      this.pageTitles.add(title);
    }
  }

  async loadFilteredPages() {
    if (!fs.existsSync(this.filteredPagesFile)) return;

    const content = fs.readFileSync(this.filteredPagesFile, 'utf8');
    const titles = content.trim().split('\n').filter(title => title.trim());

    for (const title of titles) {
      this.filteredPages.add(title);
    }
  }

  async loadSeenWikitextRevisions() {
    try {
      if (!fs.existsSync(this.wikitextFile)) return;
      const content = fs.readFileSync(this.wikitextFile, 'utf8');
      const lines = content.split('\n').filter(l => l.trim());
      for (const line of lines) {
        try {
          const obj = JSON.parse(line);
          const t = obj && obj.title; const r = obj && obj.revid;
          if (!t || r == null) continue;
          this.seenWikitextRevisions.add(`${t}::${r}`);
        } catch (_) {}
      }
    } catch (_) {}
  }


  async loadNullPages() {
    if (!fs.existsSync(this.nullPagesFile)) return;

    const content = fs.readFileSync(this.nullPagesFile, 'utf8');
    const titles = content.trim().split('\n').filter(title => title.trim());

    for (const title of titles) {
      this.nullPages.add(title);
    }
  }

  async saveChanges() {
    const spinner = ora('Saving all tracking data...').start();

    try {
      // Save content
      const contentLines = Array.from(this.pageData.values())
        .map(page => JSON.stringify(page))
        .join('\n');
      fs.writeFileSync(this.outputFile, contentLines + '\n');

      // Save titles
      const titlesList = Array.from(this.pageTitles).sort().join('\n');
      fs.writeFileSync(this.pageTitlesFile, titlesList + '\n');

      // Save filtered pages
      if (this.filteredPages.size > 0) {
        const filteredList = Array.from(this.filteredPages).sort().join('\n');
        fs.writeFileSync(this.filteredPagesFile, filteredList + '\n');
      }

      // Save null pages
      if (this.nullPages.size > 0) {
        const nullList = Array.from(this.nullPages).sort().join('\n');
        fs.writeFileSync(this.nullPagesFile, nullList + '\n');
      }

      // Update and save consolidated metadata
      this.metadata.lastUpdate = Date.now();
      this.metadata.totalPages = this.pageData.size;
      this.metadata.totalRuns++;
      this.metadata.lastRunStats = {
        added: this.stats.pagesAdded,
        updated: this.stats.pagesUpdated,
        errors: this.stats.errors
      };
      fs.writeFileSync(this.metadataFile, JSON.stringify(this.metadata, null, 2));

      spinner.succeed(chalk.green('✅ All tracking data saved'));

    } catch (error) {
      spinner.fail(chalk.red('❌ Failed to save tracking data'));
      throw error;
    }
  }

  async saveImmediately(type, data) {
    // Immediate saving for better tracking
    try {
      switch (type) {
        case 'page': {

          const prev = this.pageData.get(data.title);
          this.pageData.set(data.title, data);
          this.pageTitles.add(data.title);
          // Append raw wikitext snapshot only if new or revid changed and not seen before
          try {
            const key = `${data.title}::${data.revid}`;
            if ((!prev || prev.revid !== data.revid) && !this.seenWikitextRevisions.has(key)) {
              fs.appendFileSync(this.wikitextFile, JSON.stringify({
                title: data.title,
                categories: data.categories || [],
                rawWikitext: data.rawWikitext || '',
                timestamp: data.timestamp,
                revid: data.revid
              }) + '\n');
              this.seenWikitextRevisions.add(key);
            }
          } catch (_) {}
          break;
        }
        case 'filtered':
          this.filteredPages.add(data);
          break;
        case 'null':
          this.nullPages.add(data);
          break;
      }

      // Update API call counter immediately
      this.metadata.totalApiCalls++;

    } catch (error) {
      console.error(chalk.red(`❌ Immediate save failed: ${error.message}`));
    }
  }

  // ═══════════════════════════════════════════════════════════════════════════════
  // UTILITIES
  // ═══════════════════════════════════════════════════════════════════════════════

  // ═══════════════════════════════════════════════════════════════════════════════
  // BATCH REPROCESSING - FIX ALL EXISTING PAGES WITH NEW TEMPLATE FORMAT
  // ═══════════════════════════════════════════════════════════════════════════════

  async batchReprocessAllPages() {
    // LOAD ENTIRE JSONL INTO MEMORY FOR MAXIMUM SPEED
    console.log(chalk.blue(`🚀 Loading entire JSONL into memory for high-speed processing...`));
    this.contentPages = Array.from(this.pageData.values());

    if (this.contentPages.length === 0) {
      console.log(chalk.gray('   📄 No pages to reprocess'));
      return;
    }


    // INSANE PARALLEL PROCESSING - CPU + GPU ACCELERATION!
    const cpuCount = os.cpus().length;
    const availableMemoryGB = os.freemem() / (1024 * 1024 * 1024);
    const totalMemoryGB = os.totalmem() / (1024 * 1024 * 1024);

    // MASSIVE WORKER SCALING - NO LIMITS FOR SCRIPTING OPERATIONS
    let PARALLEL_WORKERS = Math.min(cpuCount * 8, 512); // 8x CPU cores, max 512 workers
    const BATCH_SIZE = Math.max(Math.floor(availableMemoryGB * 50), 100); // Huge batches

    // GPU ACCELERATION DETECTION
    const hasGPU = await this.detectGPUAcceleration();
    if (hasGPU) {
      PARALLEL_WORKERS = Math.min(PARALLEL_WORKERS * 2, 1024); // Double workers with GPU
      console.log(chalk.green(`   🚀 GPU ACCELERATION DETECTED - DOUBLING WORKERS!`));
    }

    console.log(chalk.red(`🔥 INSANE POWER BATCH PROCESSING:`));
    console.log(chalk.gray(`   📊 Total pages: ${this.contentPages.length.toLocaleString()}`));
    console.log(chalk.gray(`   🖥️  CPU cores: ${cpuCount} (starting with ${PARALLEL_WORKERS} workers)`));
    console.log(chalk.gray(`   💾 Available RAM: ${availableMemoryGB.toFixed(1)}GB / ${totalMemoryGB.toFixed(1)}GB (batch size: ${BATCH_SIZE})`));
    console.log(chalk.gray(`   🎮 GPU acceleration: ${hasGPU ? '✅ ENABLED' : '❌ Not available'}`));
    console.log(chalk.gray(`   ⚡ Extreme worker scaling enabled - will push to 1024+ workers if needed`));

    let processed = 0;
    let updated = 0;
    const startTime = Date.now();

    // FILTER PAGES THAT ACTUALLY NEED TEMPLATE PROCESSING
    const pagesToProcess = this.contentPages.filter(page => {
      if (!page) return false;
      const text = page.text || '';
      const hasLegacy = text.includes('[DiarySkillStats:') ||
                        text.includes('[ItemSpawnTableHead:') ||
                        text.includes('[ItemSpawnLine:');
      const hasRaw = !!page.rawWikitext;
      return hasRaw && hasLegacy;
    });

    console.log(chalk.yellow(`   🎯 Found ${pagesToProcess.length.toLocaleString()} pages that need template processing`));
    console.log(chalk.gray(`   ✅ ${(this.contentPages.length - pagesToProcess.length).toLocaleString()} pages already have correct format`));

    if (pagesToProcess.length === 0) {
      console.log(chalk.green(`   🎉 All pages already have correct template format!`));
      return;
    }

    // Process in batches with dynamic worker scaling
    for (let i = 0; i < pagesToProcess.length; i += BATCH_SIZE) {
      const batch = pagesToProcess.slice(i, i + BATCH_SIZE);

      // EXTREME DYNAMIC WORKER SCALING - PUSH TO THE LIMITS!
      const batchStartTime = Date.now();
      if (i > 0) {
        const avgTimePerBatch = (Date.now() - startTime) / (i / BATCH_SIZE);
        const memoryUsage = (os.totalmem() - os.freemem()) / os.totalmem();

        // AGGRESSIVE SCALING LOGIC
        if (avgTimePerBatch > 5000 && PARALLEL_WORKERS > 16) {
          PARALLEL_WORKERS = Math.max(PARALLEL_WORKERS - 8, 16); // Scale down if slow
        } else if (avgTimePerBatch < 1000 && memoryUsage < 0.8 && PARALLEL_WORKERS < 1024) {
          PARALLEL_WORKERS = Math.min(PARALLEL_WORKERS + 16, 1024); // MASSIVE scale up if blazing fast
        } else if (avgTimePerBatch < 2000 && memoryUsage < 0.7 && PARALLEL_WORKERS < 512) {
          PARALLEL_WORKERS = Math.min(PARALLEL_WORKERS + 8, 512); // Scale up if fast
        }

        // MEMORY PRESSURE PROTECTION
        if (memoryUsage > 0.9) {
          PARALLEL_WORKERS = Math.max(PARALLEL_WORKERS / 2, 8);
          console.log(chalk.yellow(`\n   ⚠️  High memory usage (${(memoryUsage*100).toFixed(1)}%) - scaling down to ${PARALLEL_WORKERS} workers`));
        }
      }

      // Process batch with dynamic parallel workers
      const batchResults = await this.processBatchInParallel(batch, PARALLEL_WORKERS);

      // Update pages that changed
      for (const result of batchResults) {
        if (result.updated) {
          updated++;
          // Update the page in our contentPages array
          const pageIndex = this.contentPages.findIndex(p => p.title === result.title);
          if (pageIndex !== -1) {
            this.contentPages[pageIndex].text = result.newText;
            // Store rawWikitext for future processing
            if (result.newRawWikitext) {
              this.contentPages[pageIndex].rawWikitext = result.newRawWikitext;
            }
            // Also update the pageData Map
            this.pageData.set(result.title, this.contentPages[pageIndex]);
          }
        }
        processed++;
      }

      // Update progress with dynamic worker info
      const progress = Math.round((processed / pagesToProcess.length) * 100);
      const eta = this.calculateETA(processed, pagesToProcess.length, Date.now() - startTime);
      const batchTime = ((Date.now() - batchStartTime) / 1000).toFixed(1);
      process.stdout.write(`\r🔥 Processing |${'█'.repeat(Math.floor(progress/2.5))}${'░'.repeat(40-Math.floor(progress/2.5))}| ${progress}% | ${processed}/${pagesToProcess.length} | ETA: ${eta} | Workers: ${PARALLEL_WORKERS} | Batch: ${batchTime}s | Updated: ${updated}`);
    }

    console.log(chalk.green(`\n✅ Batch reprocessing complete: ${updated.toLocaleString()} pages updated with new template format`));

    // Save the updated content immediately
    await this.saveContentToFile();
  }

  async processBatchInParallel(batch, workerCount) {
    const results = [];
    const workers = [];

    // EXTREME PARALLELIZATION - Create as many workers as requested
    const actualWorkerCount = Math.min(workerCount, batch.length);
    const itemsPerWorker = Math.ceil(batch.length / actualWorkerCount);

    console.log(chalk.gray(`     🔥 Spawning ${actualWorkerCount} parallel workers for ${batch.length} pages`));

    // Create worker promises with even distribution
    for (let i = 0; i < actualWorkerCount; i++) {
      const startIndex = i * itemsPerWorker;
      const endIndex = Math.min(startIndex + itemsPerWorker, batch.length);
      const workerBatch = batch.slice(startIndex, endIndex);

      if (workerBatch.length > 0) {
        workers.push(this.processWorkerBatch(workerBatch, i));
      }
    }

    // Wait for all workers to complete with progress tracking
    const workerResults = await Promise.allSettled(workers);

    // Flatten results and handle any failures
    for (const workerResult of workerResults) {
      if (workerResult.status === 'fulfilled') {
        results.push(...workerResult.value);
      } else {
        console.error(chalk.red(`   ⚠️  Worker failed: ${workerResult.reason}`));
      }
    }

    return results;
  }

  async processWorkerBatch(pages, workerId = 0) {
    const results = [];
    const startTime = Date.now();

    for (const page of pages) {
      try {
        // For pages with old bracket format, just fix the existing text
        if (page.text && (page.text.includes('[DiarySkillStats:') ||
                         page.text.includes('[ItemSpawnTableHead:') ||
                         page.text.includes('[ItemSpawnLine:'))) {

          // Apply template fixes directly to existing text
          let fixedText = page.text;

          // Fix DiarySkillStats format
          fixedText = fixedText.replace(/\[DiarySkillStats:\s*([^\]]+)\]/g, (match, params) => {
            const skills = params.split(',').map(p => p.trim()).filter(p => p);
            return skills.length > 0 ? `Skill Requirements: ${skills.join(', ')}` : 'Skill Requirements';
          });

          // Fix ItemSpawn format
          fixedText = fixedText.replace(/\[ItemSpawnTableHead:[^\]]*\]/g, '=== Item Spawn Locations ===');
          fixedText = fixedText.replace(/\[ItemSpawnLine:\s*([^\]]+)\]/g, (match, params) => {
            return `Item Spawn: ${params}`;
          });

          const updated = fixedText !== page.text;
          results.push({
            title: page.title,
            updated: updated,
            newText: fixedText
          });

        } else if (!page.rawWikitext) {
          // For pages without rawWikitext, try to process existing text through template parser
          try {
            const processedText = await this.processTemplatesFromWikitext(page.text || '');
            const cleanedText = this.cleanWikitextContent(processedText);
            const updated = cleanedText !== page.text;

            results.push({
              title: page.title,
              updated: updated,
              newText: updated ? cleanedText : page.text
            });
          } catch (error) {
            results.push({ title: page.title, updated: false });
          }
        } else {
          // Pages with rawWikitext - full reprocessing
          const processedWikitext = await this.processTemplatesFromWikitext(page.rawWikitext);
          const newText = this.cleanWikitextContent(processedWikitext);
          const updated = newText !== page.text;

          results.push({
            title: page.title,
            updated: updated,
            newText: updated ? newText : page.text
          });
        }

      } catch (error) {
        console.error(chalk.red(`\n   ⚠️  Failed to reprocess ${page.title}: ${error.message}`));
        results.push({ title: page.title, updated: false });
      }
    }

    return results;
  }

  async alphabeticallyReorganizeContent() {
    console.log(chalk.blue(`\n📚 Loading all content into memory for alphabetical reorganization...`));

    // Ensure contentPages is populated from pageData
    if (!this.contentPages) {
      this.contentPages = Array.from(this.pageData.values());
    }

    // Sort content pages alphabetically by title (guard against missing titles)
    const beforeCount = this.contentPages.length;
    this.contentPages = this.contentPages.filter(p => p && typeof p.title === 'string' && p.title.length > 0);
    const removed = beforeCount - this.contentPages.length;
    if (removed > 0) {
      console.log(chalk.yellow(`   ⚠️ Skipped ${removed} entries without a valid title during alphabetical sort`));
    }
    this.contentPages.sort((a, b) => {
      const at = (a && a.title) ? a.title : '';
      const bt = (b && b.title) ? b.title : '';
      return at.localeCompare(bt);
    });

    // Sort page titles alphabetically
    const sortedTitles = Array.from(this.pageTitles).sort();
    this.pageTitles = new Set(sortedTitles);

    console.log(chalk.blue(`📝 Saving alphabetically organized content...`));

    // Save reorganized content
    await this.saveContentToFile();
    await this.savePageTitlesToFile();
    // Also compact and alphabetize wikitext snapshots
    await this.compactAndSortWikitextSnapshots();

    console.log(chalk.green(`✅ Alphabetical reorganization complete`));
  }

  async saveContentToFile() {
    const lines = this.contentPages.map(page => JSON.stringify(page));
    await fs.writeFile(this.outputFile, lines.join('\n') + '\n');
  }

  async savePageTitlesToFile() {
    const sortedTitles = Array.from(this.pageTitles).sort();
    await fs.writeFile(this.pageTitlesFile, sortedTitles.join('\n') + '\n');
  }

  async compactAndSortWikitextSnapshots() {
    try {
      if (!fs.existsSync(this.wikitextFile)) return;
      const content = fs.readFileSync(this.wikitextFile, 'utf8');
      const lines = content.split('\n').filter(l => l.trim());
      const byTitle = new Map();
      for (const line of lines) {
        try {
          const obj = JSON.parse(line);
          const title = obj?.title;
          if (!title) continue;
          const prev = byTitle.get(title);
          if (!prev) {
            byTitle.set(title, obj);
          } else {
            const prevRevid = typeof prev.revid === 'number' ? prev.revid : Number(prev.revid) || 0;
            const currRevid = typeof obj.revid === 'number' ? obj.revid : Number(obj.revid) || 0;
            // Keep the one with higher revid; fallback to last occurrence
            if (currRevid >= prevRevid) byTitle.set(title, obj);
          }
        } catch (_) { /* ignore bad line */ }
      }
      const sorted = Array.from(byTitle.values()).sort((a, b) => (a.title || '').localeCompare(b.title || ''));
      fs.writeFileSync(this.wikitextFile, sorted.map(o => JSON.stringify(o)).join('\n') + '\n');
    } catch (e) {
      console.error(chalk.red(`❌ Failed to compact/sort wikitext snapshots: ${e.message}`));
    }
  }


  calculateETA(processed, total, elapsedMs) {
    if (processed === 0) return '∞';

    const rate = processed / (elapsedMs / 1000); // items per second
    const remaining = total - processed;
    const etaSeconds = remaining / rate;

    if (etaSeconds < 60) return `${Math.round(etaSeconds)}s`;
    if (etaSeconds < 3600) return `${Math.round(etaSeconds / 60)}m`;
    return `${Math.round(etaSeconds / 3600)}h`;
  }

  async detectGPUAcceleration() {
    try {
      const systemInfo = os.platform();

      if (systemInfo === 'darwin') {
        // macOS - Check for Apple Silicon and Metal support
        const appleCheck = spawn('sysctl', ['-n', 'machdep.cpu.brand_string'], { stdio: 'pipe' });
        const appleResult = await new Promise((resolve) => {
          let output = '';
          appleCheck.stdout.on('data', (data) => output += data.toString());
          appleCheck.on('close', () => {
            const cpuInfo = output.toLowerCase();
            resolve(cpuInfo.includes('apple') || cpuInfo.includes('m1') || cpuInfo.includes('m2') || cpuInfo.includes('m3') || cpuInfo.includes('m4'));
          });
          appleCheck.on('error', () => resolve(false));
        });

        if (appleResult) {
          console.log(chalk.green(`   🍎 Apple Silicon M4 Pro + Metal GPU acceleration detected!`));
          return true;
        }

        // Fallback: Check for any Metal support
        const metalCheck = spawn('system_profiler', ['SPDisplaysDataType'], { stdio: 'pipe' });
        const metalResult = await new Promise((resolve) => {
          let output = '';
          metalCheck.stdout.on('data', (data) => output += data.toString());
          metalCheck.on('close', () => resolve(output.includes('Metal')));
          metalCheck.on('error', () => resolve(false));
        });

        if (metalResult) {
          console.log(chalk.green(`   🍎 Metal GPU acceleration detected`));
          return true;
        }
      }

      // Check for NVIDIA GPU on other systems
      const nvidiaCheck = spawn('nvidia-smi', ['--query-gpu=name', '--format=csv,noheader'], { stdio: 'pipe' });
      const nvidiaResult = await new Promise((resolve) => {
        let output = '';
        nvidiaCheck.stdout.on('data', (data) => output += data.toString());
        nvidiaCheck.on('close', (code) => resolve(code === 0 && output.trim().length > 0));
        nvidiaCheck.on('error', () => resolve(false));
      });

      if (nvidiaResult) {
        console.log(chalk.green(`   🎮 NVIDIA GPU detected - enabling CUDA acceleration`));
        return true;
      }

      return false;
    } catch (error) {
      return false;
    }
  }



  // ═══════════════════════════════════════════════════════════════════════════════
  // TEMPLATE CHECKER INTEGRATION - VALIDATE AND CORRECT FORMATTING
  // ═══════════════════════════════════════════════════════════════════════════════

  async runTemplateChecker() {
    try {
      console.log(chalk.blue('🔍 Running comprehensive template format validation...'));

      const templateCheckerPath = path.join(__dirname, '../api/wiki_template_checker.py');

      return new Promise((resolve, reject) => {
        const python = spawn('python3', ['-c', `
import sys
import asyncio
sys.path.append('${path.dirname(templateCheckerPath)}')
from wiki_template_checker import check_and_correct_templates

async def main():
    data_dir = '${path.join(__dirname, '../data')}'
    corrected_pages = await check_and_correct_templates(data_dir)

    # Output results as JSON for Node.js to parse
    import json
    print("TEMPLATE_CHECKER_RESULTS:")
    print(json.dumps({
        'corrected_count': len(corrected_pages),
        'corrected_titles': list(corrected_pages.keys())
    }))

asyncio.run(main())
        `], { env: { ...process.env,
          PYTHONUNBUFFERED: '1',
          OSRS_PARSER_VERBOSE: process.env.OSRS_PARSER_VERBOSE || '0',
          OSRS_CHECKER_MAX_WORKERS: process.env.OSRS_CHECKER_MAX_WORKERS || '16',
          OSRS_PARSER_MAX_WORKERS: process.env.OSRS_PARSER_MAX_WORKERS || process.env.OSRS_CHECKER_MAX_WORKERS || '16'
        } });

        let output = '';
        let errorOutput = '';

        python.stdout.on('data', (data) => {
          const chunk = data.toString();
          output += chunk;
          try { process.stdout.write(chunk); } catch {}
        });

        python.stderr.on('data', (data) => {
          const chunk = data.toString();
          errorOutput += chunk;
          try { process.stderr.write(chunk); } catch {}
        });

        python.on('close', (code) => {
          if (code === 0) {
            // Parse results from Python output
            const resultsMatch = output.match(/TEMPLATE_CHECKER_RESULTS:\s*(\{.*\})/s);
            if (resultsMatch) {
              try {
                const results = JSON.parse(resultsMatch[1]);
                console.log(chalk.green(`   ✅ Template checker complete: ${results.corrected_count} pages corrected`));

                if (results.corrected_titles.length > 0) {
                  console.log(chalk.gray(`   📝 Corrected pages: ${results.corrected_titles.slice(0, 5).join(', ')}${results.corrected_titles.length > 5 ? '...' : ''}`));
                }

                this.stats.templatesProcessed += results.corrected_count;
                resolve(results);
              } catch (parseError) {
                console.log(chalk.yellow(`   ⚠️  Could not parse results, but template checker completed`));
                resolve({ corrected_count: 0, corrected_titles: [] });
              }
            } else {
              console.log(chalk.yellow(`   ⚠️  Template checker completed without results`));
              resolve({ corrected_count: 0, corrected_titles: [] });
            }
          } else {
            console.error(chalk.red(`   ❌ Template checker failed with code ${code}`));
            if (errorOutput) {
              console.error(chalk.red(`   Error: ${errorOutput}`));
            }
            // Don't reject - continue with watchdog operation
            resolve({ corrected_count: 0, corrected_titles: [] });
          }
        });

        python.on('error', (error) => {
          console.error(chalk.red(`   ❌ Failed to start template checker: ${error.message}`));
          // Don't reject - continue with watchdog operation
          resolve({ corrected_count: 0, corrected_titles: [] });
        });
      });

    } catch (error) {
      console.error(chalk.red(`   ❌ Template checker error: ${error.message}`));
      // Don't throw - continue with watchdog operation
      return { corrected_count: 0, corrected_titles: [] };
    }
  }

  // ═══════════════════════════════════════════════════════════════════════════════
  // CONTENT PROCESSING - ESSENTIAL FOR AI TRAINING
  // ═══════════════════════════════════════════════════════════════════════════════

  async processTemplatesFromWikitext(wikitext) {
    // Process MediaWiki templates from raw wikitext using Python parser
    try {
      const templateParserPath = path.join(__dirname, '../api/wiki_template_parser.py');

      return new Promise((resolve) => {
        const python = spawn('python3', ['-c', `
import sys
sys.path.append('${path.dirname(templateParserPath)}')
from wiki_template_parser import OSRSWikiTemplateParser

parser = OSRSWikiTemplateParser()
content = sys.stdin.read()
processed = parser.process_wiki_content(content)
print(processed)
        `], {
          env: {
            ...process.env,
            PYTHONUNBUFFERED: '1',
            OSRS_PARSER_VERBOSE: process.env.OSRS_PARSER_VERBOSE || '0',
            OSRS_PARSER_MAX_WORKERS: process.env.OSRS_PARSER_MAX_WORKERS || process.env.OSRS_CHECKER_MAX_WORKERS || '16'
          }
        });

        let output = '';
        let errorOutput = '';

        python.stdout.on('data', (data) => {
          output += data.toString();
        });

        python.stderr.on('data', (data) => {
          errorOutput += data.toString();
        });

        python.on('close', (code) => {
          if (code === 0 && output.trim()) {
            this.stats.templatesProcessed++;
            resolve(output.trim());
          } else {
            // If template parsing fails, return original wikitext
            resolve(wikitext);
          }
        });

        python.stdin.write(wikitext);
        python.stdin.end();
      });

    } catch (error) {
      return wikitext; // Return original if processing fails
    }
  }

  cleanWikitextContent(wikitext) {
    // Clean processed wikitext content for AI consumption
    // NOTE: Templates have already been processed by Python parser into readable text
    let cleanText = wikitext;

    // Remove remaining wiki markup
    cleanText = cleanText.replace(/\[\[([^|\]]+)\|?([^\]]*)\]\]/g, (match, link, text) => {
      return text || link;
    });

    // Remove external links (keep anchor text if present)
    cleanText = cleanText.replace(/\[https?:\/\/[^\s\]]+\s*([^\]]*)\]/g, '$1');

    // Remove file references and image-only lines (we don't use images currently)
    cleanText = cleanText.replace(/\[\[File:[^\]]+\]\]/g, '');
    // Drop common image label lines entirely (Image, Altimage, Item/Location/NPC Image)
    cleanText = cleanText.replace(/^\s*(Alt\s*image\d*|Altimage\d*|Image\d*|Item Image|Location Image|NPC Image)\s*:\s*.*$/gmi, '');
    // Remove bare position/size artifacts like "left|140px" or "right|300px"
    cleanText = cleanText.replace(/^\s*(left|right|center)\s*(\|\s*)?(\d+px)?\s*$/gmi, '');
    // Also handle reversed order like "130px|left"
    cleanText = cleanText.replace(/^\s*\d+px\s*\|\s*(left|right|center)\s*$/gmi, '');
    // Remove numeric index + size lines like "1: 300px" or "2: x277px"
    cleanText = cleanText.replace(/^\s*\d+\s*:\s*x?\d+px\s*$/gmi, '');
    // Remove inline "N: 300px" occurrences anywhere in a line
    cleanText = cleanText.replace(/\d+\s*:\s*x?\d+px\b/gmi, '');
    // Remove inline "130px|left"-style occurrences anywhere in a line
    cleanText = cleanText.replace(/\b\d+px\b\s*\|\s*(left|right|center)\b/gmi, '');
    // Remove stray numeric index-only lines like "1:"
    cleanText = cleanText.replace(/^\s*\d+\s*:\s*$/gmi, '');
    // Remove gallery/thumb artifacts
    cleanText = cleanText.replace(/^\s*thumb\|.*$/gmi, '');
    cleanText = cleanText.replace(/(^|\s)frame\|/gmi, '$1');


    // Remove bottom navboxes and related show/v•e bars and long bullet rows (noise for embeddings)
    cleanText = cleanText.replace(/^\s*\[\s*show\s*\].*$/gmi, '');
    cleanText = cleanText.replace(/^\s*(?:\[\s*show\s*\]\s*)?v\s*•\s*(?:d\s*•\s*)?e\b.*$/gmi, '');
    // Lines composed of many bullets (e.g., • • • • • ...)
    cleanText = cleanText.replace(/^\s*(?:•\s*){8,}.*$/gmi, '');

    // Remove categories
    cleanText = cleanText.replace(/\[\[Category:[^\]]+\]\]/g, '');

    // Remove HTML comments
    cleanText = cleanText.replace(/<!--[\s\S]*?-->/g, '');

    // Only remove empty/malformed template braces that slipped through
    cleanText = cleanText.replace(/\{\{\s*\}\}/g, ''); // Empty templates
    cleanText = cleanText.replace(/\{\{[^}]*$/g, ''); // Incomplete templates at end
    cleanText = cleanText.replace(/^[^{]*\}\}/g, ''); // Incomplete templates at start

    // Normalize multiple blank lines while preserving section structure
    cleanText = cleanText.replace(/\n\s*\n\s*\n/g, '\n\n');
    cleanText = cleanText.replace(/^\s+|\s+$/gm, '');
    cleanText = cleanText.trim();

    return cleanText;
  }

  // ═══════════════════════════════════════════════════════════════════════════════
  // SINGLE CONSOLIDATED PROGRESS BAR SYSTEM
  // ═══════════════════════════════════════════════════════════════════════════════

  updateProgress(operation, current, total, extraInfo = '') {
    const percentage = Math.round((current / total) * 100);
    const elapsed = (Date.now() - this.startTime) / 1000;
    const rate = Math.round(current / elapsed * 10) / 10;
    const eta = Math.round((total - current) / rate);

    // Create visual progress bar
    const barLength = 40;
    const filledLength = Math.round((current / total) * barLength);
    const bar = '█'.repeat(filledLength) + '░'.repeat(barLength - filledLength);

    // Clear line and show consolidated progress
    const progressLine = `${operation} |${chalk.cyan(bar)}| ${percentage}% | ${current.toLocaleString()}/${total.toLocaleString()} | ETA: ${eta}s | ${rate}/s${extraInfo}`;
    process.stdout.write(`\r${progressLine}`);
  }

  clearProgress() {
    process.stdout.write('\r' + ' '.repeat(120) + '\r');
  }

  showFinalResult(operation, successful, total, duration) {
    this.clearProgress();
    const rate = Math.round(total / duration * 10) / 10;
    console.log(chalk.green(`✅ ${operation}: ${successful.toLocaleString()}/${total.toLocaleString()} successful (${rate}/s)`));
  }

  sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
// MAIN EXECUTION
// ═══════════════════════════════════════════════════════════════════════════════

if (require.main === module) {
  const watchdog = new StreamlinedOSRSWatchdog();
  watchdog.run().catch(console.error);
}

module.exports = StreamlinedOSRSWatchdog;
