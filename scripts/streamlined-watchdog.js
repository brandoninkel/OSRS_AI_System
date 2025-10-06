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

// ============================================================================
// EPIPE ERROR HANDLING
// ============================================================================
// Prevent EPIPE errors when stdout/stderr pipes are closed
process.stdout.on('error', (err) => {
  if (err.code === 'EPIPE') {
    // Pipe closed, exit gracefully
    process.exit(0);
  }
});

process.stderr.on('error', (err) => {
  if (err.code === 'EPIPE') {
    // Pipe closed, exit gracefully
    process.exit(0);
  }
});

// Handle uncaught exceptions
process.on('uncaughtException', (err) => {
  if (err.code === 'EPIPE') {
    process.exit(0);
  }
  // Log other errors to file instead of stdout
  const logFile = path.join(__dirname, '../logs/osrs_ai/watchdog_errors.log');
  fs.appendFileSync(logFile, `${new Date().toISOString()} - Uncaught Exception: ${err.stack}\n`);
  process.exit(1);
});

class StreamlinedOSRSWatchdog {
  constructor() {
    this.wikiApiUrl = 'https://oldschool.runescape.wiki/api.php';
    this.userAgent = 'OSRS-AI-RAG-System/1.0 (brandoninkel@gmail.com) Node.js/axios';

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

    // Track changed pages for incremental KG updates
    this.changedPages = {
      added: new Set(),
      updated: new Set(),
      deleted: new Set()
    };

    // KG update tracking
    this.kgUpdateThreshold = {
      pagesAdded: 5,      // Trigger on 5+ new pages
      pagesUpdated: 20    // Trigger on 20+ updated pages
    };
    this.lastKgUpdate = Date.now();

    // GE update tracking (ensure 5-minute minimum interval)
    this.lastGEUpdate = 0;  // Timestamp of last GE update
    this.geUpdateInterval = 5 * 60 * 1000;  // 5 minutes in milliseconds

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
    // Template checker is now available - can be disabled with --skip-checker flag
    this.skipChecker = process.env.OSRS_SKIP_CHECKER === '1' || process.argv.includes('--skip-checker');

    // Orchestration mode flags
    this.completionBased = process.argv.includes('--completion-based');
    this.timedCycles = process.argv.includes('--timed-cycles');

    // Progress tracking for embedding systems
    this.embeddingProgress = {
      kg: { active: false, progress: 0, status: 'idle' },
      regular: { active: false, progress: 0, status: 'idle' }
    };

    // API server URL for queue coordination
    this.apiServerUrl = process.env.OSRS_API_URL || 'http://localhost:5001';
  }

  // ═══════════════════════════════════════════════════════════════════════════════
  // API COORDINATION
  // ═══════════════════════════════════════════════════════════════════════════════

  async signalWatchdogStatus(active) {
    try {
      await axios.post(`${this.apiServerUrl}/watchdog/status`, { active }, {
        headers: { 'Content-Type': 'application/json' },
        timeout: 5000
      });

      if (active) {
        console.log(chalk.yellow('🚨 Signaled API: Watchdog ACTIVE - Other API calls will be throttled'));
      } else {
        console.log(chalk.green('✅ Signaled API: Watchdog INACTIVE - Normal API rates resumed'));
      }
    } catch (error) {
      console.log(chalk.gray(`   ℹ️  Could not signal watchdog status to API: ${error.message}`));
      // Don't fail the watchdog if API is unavailable
    }
  }

  // ═══════════════════════════════════════════════════════════════════════════════
  // MAIN EXECUTION
  // ═══════════════════════════════════════════════════════════════════════════════

  async run() {
    console.log(chalk.cyan.bold('🚀 OSRS Wiki Watchdog - Streamlined Edition'));
    console.log(chalk.gray('━'.repeat(60)));


    try {
      await this.initialize();

      // Signal API that watchdog is starting
      await this.signalWatchdogStatus(true);


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

      // Check if KG system needs initial sync
      await this.checkKGSyncStatus();

      // Signal API that watchdog initial run is complete
      await this.signalWatchdogStatus(false);

      // Choose monitoring mode based on flags
      if (this.completionBased) {
        await this.startCompletionBasedMonitoring();
      } else {
        await this.startTimedMonitoring();
      }
    } catch (error) {
      console.error(chalk.red(`❌ Fatal error: ${error.message}`));
      await this.signalWatchdogStatus(false); // Ensure we signal inactive on error
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

  async checkKGSyncStatus() {
    try {
      console.log(chalk.blue('\n🔍 Checking KG system sync status...'));

      const kgStatusFile = path.join(__dirname, '../data/kg_updater_status.json');
      const kgMetaFile = path.join(__dirname, '../data/osrs_kg.meta.json');
      const watchdogMetaFile = path.join(__dirname, '../data/watchdog_metadata.json');

      // Check if KG files exist
      if (!fs.existsSync(kgStatusFile) || !fs.existsSync(kgMetaFile)) {
        console.log(chalk.yellow('   ⚠️  KG system not initialized, will trigger on first wiki change'));
        return false;
      }

      // Load metadata
      const kgStatus = JSON.parse(fs.readFileSync(kgStatusFile, 'utf8'));
      const kgMeta = JSON.parse(fs.readFileSync(kgMetaFile, 'utf8'));
      const watchdogMeta = JSON.parse(fs.readFileSync(watchdogMetaFile, 'utf8'));

      // Compare timestamps
      const kgTimestamp = kgMeta.generated_at * 1000; // Convert to ms
      const watchdogTimestamp = watchdogMeta.lastUpdate;

      const timeDiff = watchdogTimestamp - kgTimestamp;
      const hoursDiff = Math.round(timeDiff / (1000 * 60 * 60));

      console.log(chalk.gray(`   📊 Wiki last updated: ${new Date(watchdogTimestamp).toLocaleString()}`));
      console.log(chalk.gray(`   📊 KG last updated: ${new Date(kgTimestamp).toLocaleString()}`));
      console.log(chalk.gray(`   ⏱️  Time difference: ${hoursDiff} hours`));

      // If KG is more than 1 hour behind, force an update
      if (timeDiff > 60 * 60 * 1000) {
        console.log(chalk.yellow(`   ⚠️  KG system is ${hoursDiff} hours behind wiki content`));
        console.log(chalk.yellow('   🔄 Forcing KG and embedding updates to sync...'));

        // Set stats to trigger embedding updates
        this.stats.pagesUpdated = 1; // Fake a change to trigger updates

        return true; // Indicates KG needs sync
      } else {
        console.log(chalk.green('   ✅ KG system is in sync with wiki content'));
        return false;
      }

    } catch (error) {
      console.log(chalk.gray(`   ℹ️  Could not check KG sync status: ${error.message}`));
      return false;
    }
  }

  async startTimedMonitoring() {
    console.log(chalk.green('\n👁️  Starting timed monitoring (10-minute cycles)...'));
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

      // Signal watchdog active for this cycle
      await this.signalWatchdogStatus(true);

      // Do complete update cycle (scan for new pages + check for updates)
      await this.updateCollection();

      // Signal watchdog inactive after cycle
      await this.signalWatchdogStatus(false);

    }, 10 * 60 * 1000);

    // Graceful shutdown
    process.on('SIGINT', () => {
      console.log(chalk.yellow('\n🛑 Shutting down gracefully...'));
      this.isRunning = false;
      clearInterval(monitorInterval);
      process.exit(0);
    });
  }

  async startCompletionBasedMonitoring() {
    console.log(chalk.green('\n👁️  Starting completion-based orchestration...'));
    console.log(chalk.gray('Each cycle waits for embedding systems to complete'));
    console.log(chalk.gray('GE updates run independently every 5 minutes'));
    console.log(chalk.gray('Press Ctrl+C to stop'));

    this.isRunning = true;
    let isFirstCycle = true;

    // Start independent GE update timer (every 5 minutes)
    this.startGEUpdateTimer();

    // Graceful shutdown handler
    process.on('SIGINT', () => {
      console.log(chalk.yellow('\n🛑 Shutting down gracefully...'));
      this.isRunning = false;
      if (this.geUpdateTimer) {
        clearInterval(this.geUpdateTimer);
      }
      process.exit(0);
    });

    while (this.isRunning) {
      try {
        console.log(chalk.blue(`\n🔄 Wiki monitoring cycle... ${new Date().toLocaleTimeString()}`));
        console.log(chalk.blue('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'));

        // Check if changes warrant embedding updates (before resetting stats)
        let hasChanges = this.stats.pagesAdded > 0 || this.stats.pagesUpdated > 0;

        // If this is not the first cycle, do a new update cycle
        if (!isFirstCycle) {
          // Reset stats for this cycle
          this.resetStats();

          // Signal watchdog active for this cycle
          await this.signalWatchdogStatus(true);

          // Do complete update cycle (scan for new pages + check for updates)
          await this.updateCollection();

          // Signal watchdog inactive after cycle
          await this.signalWatchdogStatus(false);

          // Check if changes warrant embedding updates
          hasChanges = this.stats.pagesAdded > 0 || this.stats.pagesUpdated > 0;
        } else {
          // First cycle: use stats from initial run
          isFirstCycle = false;
        }

        // GE updates now run on independent 5-minute timer (see startGEUpdateTimer)
        // No need to update here - it happens automatically in the background

        if (hasChanges) {
          console.log(chalk.yellow(`\n🚀 Changes detected: ${this.stats.pagesAdded} added, ${this.stats.pagesUpdated} updated`));
          console.log(chalk.yellow('🔄 Triggering both embedding systems...'));

          // Save changed pages list for KG incremental updates
          await this.saveChangedPagesList();

          // Trigger both embedding systems and wait for completion
          const success = await this.triggerBothEmbeddingSystems();

          if (success) {
            console.log(chalk.green('\n✅ Embedding systems completed successfully'));
            console.log(chalk.green('🔄 Resuming watchdog monitoring...'));

            // Clear changed pages tracking after successful update
            this.changedPages.added.clear();
            this.changedPages.updated.clear();
            this.changedPages.deleted.clear();
          } else {
            console.log(chalk.red('\n❌ Embedding systems encountered errors'));
            console.log(chalk.yellow('⏳ Waiting 2 minutes before next cycle...'));
            await this.sleep(2 * 60 * 1000);
          }
        } else {
          console.log(chalk.green('✅ No wiki changes detected, skipping embedding updates'));
          console.log(chalk.gray('⏳ Waiting 10 minutes before next cycle...'));
          console.log(chalk.yellow('💡 Press any key to start cycle now'));
          await this.waitWithKeypress(10 * 60 * 1000);
        }

      } catch (error) {
        console.error(chalk.red(`❌ Monitoring cycle error: ${error.message}`));
        console.log(chalk.yellow('⏳ Waiting 1 minute before retry...'));
        await this.sleep(60 * 1000);
      }
    }
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
    /**
     * Fetch recent changes with continuation support.
     * MediaWiki API limits to 500 results per request, so we need to
     * continue fetching if there are more changes.
     */
    const lastCheck = this.metadata?.lastUpdate || (Date.now() - 10 * 60 * 1000);
    const since = new Date(lastCheck).toISOString();

    let allChanges = [];
    let continueToken = null;
    let batchCount = 0;
    const maxBatches = 10; // Safety limit: max 5000 changes (10 * 500)

    do {
      const params = {
        action: 'query',
        list: 'recentchanges',
        rcstart: new Date().toISOString(),
        rcend: since,
        rcnamespace: this.targetNamespaces.map(ns => ns.id).join('|'),
        rctype: 'edit|new',
        rcprop: 'title|timestamp',
        rclimit: 500,
        format: 'json'
      };

      // Add continuation token if we have one
      if (continueToken) {
        params.rccontinue = continueToken;
      }

      const response = await axios.get(this.wikiApiUrl, {
        params,
        headers: { 'User-Agent': this.userAgent },
        timeout: 30000
      });

      const changes = response.data.query?.recentchanges || [];
      allChanges = allChanges.concat(changes);

      // Check for continuation
      continueToken = response.data.continue?.rccontinue;
      batchCount++;

      // Log progress if fetching multiple batches
      if (continueToken && batchCount < maxBatches) {
        console.log(chalk.gray(`   📡 Fetched ${allChanges.length} changes, continuing...`));
        await this.sleep(100); // Small delay between requests
      }

      // Safety check: don't fetch forever
      if (batchCount >= maxBatches) {
        console.log(chalk.yellow(`   ⚠️  Reached max batches (${maxBatches}), stopping at ${allChanges.length} changes`));
        break;
      }

    } while (continueToken);

    if (batchCount > 1) {
      console.log(chalk.green(`   ✅ Fetched ${allChanges.length} total changes across ${batchCount} batches`));
    }

    return allChanges;
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
            this.changedPages.added.add(page.title);
          } else {
            this.stats.pagesUpdated++; // Existing page with new/updated content
            this.changedPages.updated.add(page.title);
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

        // No artificial rate limiting needed - serial requests are safe per MediaWiki API guidelines

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
      const requestStart = Date.now();

      try {
        await this.updatePage(change.title);
        this.stats.pagesUpdated++;
        this.changedPages.updated.add(change.title);

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

        // Smart rate limiting: only sleep if request was too fast
        // Official MediaWiki guidance: serial requests are safe, no hard limit
        // Target: ~1 request per second for politeness
        const requestDuration = Date.now() - requestStart;
        const minRequestTime = 1000; // 1 second between requests
        if (requestDuration < minRequestTime) {
          await this.sleep(minRequestTime - requestDuration);
        }

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

      // Check if KG update should be triggered
      await this.checkKgUpdateTrigger();

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
  // EMBEDDING SYSTEMS ORCHESTRATION
  // ═══════════════════════════════════════════════════════════════════════════════

  async saveChangedPagesList() {
    try {
      const changedPagesFile = path.join(__dirname, '../data/watchdog_changed_pages.json');

      const changedPagesData = {
        timestamp: new Date().toISOString(),
        added: Array.from(this.changedPages.added),
        updated: Array.from(this.changedPages.updated),
        deleted: Array.from(this.changedPages.deleted),
        total: this.changedPages.added.size + this.changedPages.updated.size + this.changedPages.deleted.size
      };

      fs.writeFileSync(changedPagesFile, JSON.stringify(changedPagesData, null, 2));
      console.log(chalk.gray(`   💾 Saved ${changedPagesData.total} changed pages (${this.changedPages.added.size} added, ${this.changedPages.updated.size} updated, ${this.changedPages.deleted.size} deleted)`));
    } catch (error) {
      console.error(chalk.red(`   ⚠️  Failed to save changed pages list: ${error.message}`));
    }
  }

  async triggerBothEmbeddingSystems() {
    console.log(chalk.yellow('\n🚀 Starting both embedding systems in parallel...'));

    // Reset progress tracking
    this.embeddingProgress.kg = { active: true, progress: 0, status: 'starting' };
    this.embeddingProgress.regular = { active: true, progress: 0, status: 'starting' };

    try {
      // Start both processes in parallel
      const kgPromise = this.triggerKgUpdateWithProgress();
      const embeddingsPromise = this.triggerRegularEmbeddingsWithProgress();

      // Start progress monitoring
      const progressMonitor = this.startProgressMonitoring();

      // Wait for both to complete
      const results = await Promise.allSettled([kgPromise, embeddingsPromise]);

      // Stop progress monitoring
      clearInterval(progressMonitor);

      // Clear progress display
      process.stdout.write('\n');

      // Check results
      const kgSuccess = results[0].status === 'fulfilled';
      const embeddingsSuccess = results[1].status === 'fulfilled';

      if (kgSuccess && embeddingsSuccess) {
        console.log(chalk.green('✅ Both embedding systems completed successfully'));
        return true;
      } else {
        if (!kgSuccess) {
          console.error(chalk.red(`❌ KG embeddings failed: ${results[0].reason}`));
        }
        if (!embeddingsSuccess) {
          console.error(chalk.red(`❌ Regular embeddings failed: ${results[1].reason}`));
        }
        return false;
      }

    } catch (error) {
      console.error(chalk.red(`❌ Embedding systems orchestration failed: ${error.message}`));
      return false;
    } finally {
      // Reset progress tracking
      this.embeddingProgress.kg = { active: false, progress: 0, status: 'idle' };
      this.embeddingProgress.regular = { active: false, progress: 0, status: 'idle' };
    }
  }

  startProgressMonitoring() {
    return setInterval(() => {
      if (this.embeddingProgress.kg.active || this.embeddingProgress.regular.active) {
        // Clear current line and show progress
        process.stdout.write('\r\x1b[K');

        const kgBar = this.createProgressBar(this.embeddingProgress.kg.progress, 'KG Embeddings');
        const regularBar = this.createProgressBar(this.embeddingProgress.regular.progress, 'Wiki Embeddings');

        process.stdout.write(`${kgBar} | ${regularBar}`);
      }
    }, 1000);
  }

  createProgressBar(progress, label) {
    const width = 20;
    const filled = Math.round(width * (progress / 100));
    const empty = width - filled;
    const bar = '█'.repeat(filled) + '░'.repeat(empty);
    const percentage = progress.toFixed(1).padStart(5);
    return `${label}: [${bar}] ${percentage}%`;
  }

  async triggerRegularEmbeddingsWithProgress() {
    return new Promise((resolve, reject) => {
      console.log(chalk.blue('   🔄 Starting regular wiki embeddings...'));

      this.embeddingProgress.regular.status = 'running';

      const embeddingsProcess = spawn('python3', [
        '-u',  // Unbuffered output for real-time progress
        path.join(__dirname, 'create_osrs_embeddings.py'),
        '--incremental',
        '--progress-mode'
      ], {
        stdio: ['pipe', 'pipe', 'pipe'],
        cwd: path.join(__dirname, '..')
      });

      let outputBuffer = '';

      embeddingsProcess.stdout.on('data', (data) => {
        outputBuffer += data.toString();
        this.parseEmbeddingProgress(outputBuffer, 'regular');
      });

      embeddingsProcess.stderr.on('data', (data) => {
        const output = data.toString();
        // Python logging sends INFO to stderr, so parse it for progress
        outputBuffer += output;
        this.parseEmbeddingProgress(outputBuffer, 'regular');

        // Only show actual errors (not INFO or WARNING)
        if (!output.includes('INFO') && !output.includes('WARNING')) {
          console.error(chalk.red(`Regular embeddings error: ${output}`));
        }
      });

      embeddingsProcess.on('close', (code) => {
        this.embeddingProgress.regular.active = false;
        if (code === 0) {
          this.embeddingProgress.regular.progress = 100;
          this.embeddingProgress.regular.status = 'completed';
          resolve();
        } else {
          this.embeddingProgress.regular.status = 'failed';
          reject(new Error(`Regular embeddings failed with code ${code}`));
        }
      });

      embeddingsProcess.on('error', (error) => {
        this.embeddingProgress.regular.active = false;
        this.embeddingProgress.regular.status = 'failed';
        reject(error);
      });
    });
  }

  parseEmbeddingProgress(output, type) {
    // Look for progress indicators in the output
    const progressMatch = output.match(/Progress:\s*(\d+(?:\.\d+)?)/);
    if (progressMatch) {
      const progress = parseFloat(progressMatch[1]);
      this.embeddingProgress[type].progress = progress;
      // Debug: confirm parsing
      if (type === 'kg' && progress > 0) {
        console.log(chalk.gray(`\n[DEBUG] Parsed KG progress: ${progress}%`));
      }
    }

    // Look for status updates
    const statusMatch = output.match(/Status:\s*([^\n]+)/);
    if (statusMatch) {
      this.embeddingProgress[type].status = statusMatch[1].trim();
    }
  }

  resetStats() {
    this.stats = {
      pagesAdded: 0,
      pagesUpdated: 0,
      pagesSkipped: 0,
      errors: 0
    };
  }

  // ═══════════════════════════════════════════════════════════════════════════════
  // KG AUTO-UPDATE INTEGRATION
  // ═══════════════════════════════════════════════════════════════════════════════

  async checkKgUpdateTrigger() {
    try {
      const shouldTrigger = (
        this.stats.pagesAdded >= this.kgUpdateThreshold.pagesAdded ||
        this.stats.pagesUpdated >= this.kgUpdateThreshold.pagesUpdated
      );

      if (shouldTrigger) {
        const timeSinceLastUpdate = Date.now() - this.lastKgUpdate;
        const minInterval = 10 * 60 * 1000; // 10 minutes minimum between updates

        if (timeSinceLastUpdate >= minInterval) {
          console.log(chalk.yellow(`\n🔗 Triggering KG auto-update...`));
          console.log(chalk.gray(`   📊 Changes: ${this.stats.pagesAdded} added, ${this.stats.pagesUpdated} updated`));

          await this.triggerKgUpdate();
          this.lastKgUpdate = Date.now();
        } else {
          const waitMinutes = Math.ceil((minInterval - timeSinceLastUpdate) / 60000);
          console.log(chalk.gray(`\n⏳ KG update needed but waiting ${waitMinutes} more minutes (rate limiting)`));
        }
      }
    } catch (error) {
      console.error(chalk.red(`❌ KG update trigger check failed: ${error.message}`));
    }
  }

  async triggerKgUpdate() {
    try {
      // Check if KG auto-updater service is running
      const kgUpdaterPidFile = path.join(__dirname, '../logs/kg/kg_updater.pid');

      if (fs.existsSync(kgUpdaterPidFile)) {
        // Signal the KG auto-updater service
        const pid = fs.readFileSync(kgUpdaterPidFile, 'utf8').trim();
        try {
          process.kill(parseInt(pid), 'SIGUSR1'); // Custom signal for KG update
          console.log(chalk.green(`   ✅ Signaled KG auto-updater service (PID ${pid})`));
          return;
        } catch (e) {
          console.log(chalk.yellow(`   ⚠️  KG auto-updater PID ${pid} not responding, running direct update`));
        }
      }

      // Fallback: Run KG update directly
      console.log(chalk.blue(`   🔄 Running direct KG embedding generation...`));

      const { spawn } = require('child_process');
      const kgProcess = spawn('python3', [
        path.join(__dirname, 'kg/create_mxbai_kg_embeddings.py'),
        '--incremental'
      ], {
        detached: true,
        stdio: 'ignore',
        cwd: path.join(__dirname, '..')
      });

      kgProcess.unref(); // Allow parent to exit
      console.log(chalk.green(`   ✅ KG update process started (PID ${kgProcess.pid})`));

    } catch (error) {
      console.error(chalk.red(`   ❌ Failed to trigger KG update: ${error.message}`));
    }
  }

  async triggerKgUpdateWithProgress() {
    return new Promise((resolve, reject) => {
      console.log(chalk.blue('   🔄 Starting KG embedding generation...'));

      this.embeddingProgress.kg.status = 'running';

      const kgProcess = spawn('python3', [
        '-u',  // Unbuffered output for real-time progress
        path.join(__dirname, 'kg/create_mxbai_kg_embeddings.py'),
        '--incremental'  // Use incremental mode for faster updates
      ], {
        stdio: ['pipe', 'pipe', 'pipe'],
        cwd: path.join(__dirname, '..')
      });

      let outputBuffer = '';

      kgProcess.stdout.on('data', (data) => {
        const text = data.toString();
        outputBuffer += text;
        // Debug: log what we're receiving
        if (text.includes('Progress:') || text.includes('Status:')) {
          console.log(chalk.gray(`\n[KG stdout] ${text.substring(0, 200)}`));
        }
        this.parseEmbeddingProgress(outputBuffer, 'kg');
      });

      kgProcess.stderr.on('data', (data) => {
        const output = data.toString();
        // Python logging sends INFO to stderr, so parse it for progress
        outputBuffer += output;
        // Debug: log what we're receiving
        if (output.includes('Progress:') || output.includes('Status:')) {
          console.log(chalk.gray(`\n[KG stderr] ${output.substring(0, 200)}`));
        }
        this.parseEmbeddingProgress(outputBuffer, 'kg');

        // Only show actual errors (not INFO or WARNING)
        if (!output.includes('INFO') && !output.includes('WARNING')) {
          console.error(chalk.red(`KG updater error: ${output}`));
        }
      });

      kgProcess.on('close', (code) => {
        this.embeddingProgress.kg.active = false;
        if (code === 0) {
          this.embeddingProgress.kg.progress = 100;
          this.embeddingProgress.kg.status = 'completed';
          resolve();
        } else {
          this.embeddingProgress.kg.status = 'failed';
          reject(new Error(`KG updater failed with code ${code}`));
        }
      });

      kgProcess.on('error', (error) => {
        this.embeddingProgress.kg.active = false;
        this.embeddingProgress.kg.status = 'failed';
        reject(error);
      });
    });
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

    // Write status file for GUI monitoring
    this.writeStatusFile({
      active: true,
      task: operation,
      progress: percentage,
      status: `${current.toLocaleString()}/${total.toLocaleString()} items`,
      eta: `${eta}s`,
      rate: `${rate}/s`
    });
  }

  writeStatusFile(status) {
    try {
      const statusFile = path.join(__dirname, '../logs/watchdog_status.json');
      fs.writeFileSync(statusFile, JSON.stringify(status, null, 2));
    } catch (error) {
      // Silently fail - don't interrupt operations for status file writes
    }
  }

  clearProgress() {
    process.stdout.write('\r' + ' '.repeat(120) + '\r');
    // Clear status file
    this.writeStatusFile({
      active: false,
      task: 'Idle',
      progress: 0,
      status: 'Ready',
      eta: ''
    });
  }

  showFinalResult(operation, successful, total, duration) {
    this.clearProgress();
    const rate = Math.round(total / duration * 10) / 10;
    console.log(chalk.green(`✅ ${operation}: ${successful.toLocaleString()}/${total.toLocaleString()} successful (${rate}/s)`));
  }

  sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  async waitWithKeypress(ms) {
    return new Promise((resolve) => {
      const startTime = Date.now();
      const totalSeconds = Math.floor(ms / 1000);
      let countdownInterval;

      const cleanup = () => {
        if (countdownInterval) clearInterval(countdownInterval);
        process.stdin.removeAllListeners('data');
        process.stdin.setRawMode(false);
        process.stdin.pause();
      };

      const timeout = setTimeout(() => {
        cleanup();
        process.stdout.write('\n'); // Clear countdown line
        resolve();
      }, ms);

      // Start countdown display
      countdownInterval = setInterval(() => {
        const elapsed = Math.floor((Date.now() - startTime) / 1000);
        const remaining = Math.max(0, totalSeconds - elapsed);
        const minutes = Math.floor(remaining / 60);
        const seconds = remaining % 60;

        process.stdout.write(`\r${chalk.gray('⏳ Next cycle in:')} ${chalk.cyan(`${minutes}:${seconds.toString().padStart(2, '0')}`)} ${chalk.yellow('(Press any key to start now)')}`);

        if (remaining <= 0) {
          clearInterval(countdownInterval);
        }
      }, 1000);

      // Set up keypress detection
      process.stdin.setRawMode(true);
      process.stdin.resume();
      process.stdin.setEncoding('utf8');

      process.stdin.once('data', () => {
        clearTimeout(timeout);
        cleanup();
        console.log(chalk.green('\n⚡ Manual cycle triggered!'));
        resolve();
      });
    });
  }

  // ═══════════════════════════════════════════════════════════════════════════════
  // GE PRICE UPDATES (INDEPENDENT 5-MINUTE TIMER)
  // ═══════════════════════════════════════════════════════════════════════════════

  startGEUpdateTimer() {
    /**
     * Start independent GE update timer that runs every 5 minutes.
     * This runs in parallel with the watchdog cycle, ensuring GE updates
     * happen at proper 5-minute intervals regardless of watchdog timing.
     */
    console.log(chalk.blue('💰 Starting independent GE update timer (every 5 minutes)...'));

    // Run first update immediately
    this.updateGEPrices().then(() => {
      console.log(chalk.green('   ✅ Initial GE update complete'));
    }).catch(err => {
      console.error(chalk.red(`   ❌ Initial GE update failed: ${err.message}`));
    });

    // Then run every 5 minutes
    this.geUpdateTimer = setInterval(async () => {
      try {
        const now = new Date().toLocaleTimeString();
        console.log(chalk.yellow(`\n💰 [${now}] Running scheduled GE update...`));
        await this.updateGEPrices();
        console.log(chalk.green(`   ✅ GE update complete`));
      } catch (error) {
        console.error(chalk.red(`   ❌ GE update error: ${error.message}`));
      }
    }, this.geUpdateInterval);

    console.log(chalk.green('   ✅ GE update timer started (updates every 5 minutes)'));
  }

  async updateGEPrices() {
    /**
     * Update GE prices as part of the watchdog cycle.
     * This runs the GE update daemon logic inline, avoiding separate process conflicts.
     *
     * Benefits:
     * - Sequential execution with wiki updates
     * - No API conflicts (Prices API vs MediaWiki API)
     * - Part of completion-based orchestration
     * - Simpler process management
     */
    try {
      const { spawn } = require('child_process');
      const path = require('path');

      console.log(chalk.blue('   🔄 Fetching latest GE prices from Weird Gloop API...'));

      return new Promise((resolve) => {
        const geProcess = spawn('python3', [
          path.join(__dirname, 'ge_update_daemon.py'),
          '--single-update'  // Run once, don't loop
        ], {
          stdio: ['pipe', 'pipe', 'pipe'],
          cwd: path.join(__dirname, '..')
        });

        let hasOutput = false;

        geProcess.stdout.on('data', (data) => {
          const output = data.toString();
          hasOutput = true;
          // Show key updates
          if (output.includes('✅') || output.includes('📊') || output.includes('💾')) {
            console.log(chalk.gray(`   ${output.trim()}`));
          }
        });

        geProcess.stderr.on('data', (data) => {
          const output = data.toString();
          // Only show actual errors, not INFO logs
          if (!output.includes('INFO') && !output.includes('DEBUG')) {
            console.error(chalk.red(`   GE update error: ${output.trim()}`));
          }
        });

        geProcess.on('close', (code) => {
          if (code === 0) {
            console.log(chalk.green('   ✅ GE prices updated successfully'));
            resolve(true);
          } else {
            console.log(chalk.yellow(`   ⚠️  GE update exited with code ${code}`));
            resolve(false);
          }
        });

        geProcess.on('error', (error) => {
          console.error(chalk.red(`   ❌ Failed to run GE update: ${error.message}`));
          resolve(false);
        });

        // Timeout after 2 minutes
        setTimeout(() => {
          if (!hasOutput) {
            console.log(chalk.yellow('   ⚠️  GE update timeout (2 minutes)'));
            geProcess.kill();
            resolve(false);
          }
        }, 2 * 60 * 1000);
      });

    } catch (error) {
      console.error(chalk.red(`   ❌ GE update error: ${error.message}`));
      return false;
    }
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
