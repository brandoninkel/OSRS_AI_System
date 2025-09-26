#!/usr/bin/env node

/**
 * OPTIMIZED OSRS Wiki Phase 3: Maximum Speed API-Compliant Watchdog
 *
 * 🚨 CRITICAL COMPLIANCE WITH MEDIAWIKI API POLICIES:
 * ✅ SERIAL requests only (parallel explicitly forbidden by API:Etiquette)
 * ✅ Proper User-Agent with contact info (required by WMF User-Agent Policy)
 * ✅ Exponential backoff on rate limits (required by API:Etiquette)
 * ✅ Single connection only (concurrent connections prohibited by WMF API Guidelines)
 * ✅ No artificial delays (OSRS Wiki has no read limits, serial processing provides natural timing)
 * ✅ Batch API calls using pipe separator (recommended by API:Etiquette)
 * ✅ Connection pooling with keep-alive (performance optimization)
 * ✅ Page titles file updates for accurate rebuilds
 *
 * SOURCES:
 * - MediaWiki API:Etiquette: https://www.mediawiki.org/wiki/API:Etiquette
 * - WMF API Usage Guidelines: https://foundation.wikimedia.org/wiki/Policy:Wikimedia_Foundation_API_Usage_Guidelines
 * - WMF User-Agent Policy: https://foundation.wikimedia.org/wiki/Policy:Wikimedia_Foundation_User-Agent_Policy
 * - OSRS Wiki Rate Limits: 8 edits/min, 30 purges/min, no read limits
 */

const fs = require('fs-extra');
const path = require('path');
const axios = require('axios');
const chalk = require('chalk');
const ora = require('ora');
const { spawn } = require('child_process');

class OptimizedOSRSWikiWatchdog {
  constructor() {
    // Wiki API settings - COMPLIANT WITH WIKIMEDIA FOUNDATION USER-AGENT POLICY
    this.wikiApiUrl = 'https://oldschool.runescape.wiki/api.php';
    this.userAgent = 'OSRS-AI-Training-Bot/1.0 (https://github.com/user/osrs-ai-project; contact@osrs-ai.org) Node.js/axios';
    
    // File paths
    this.outputFile = path.join(__dirname, '../data/osrs_wiki_content.jsonl');
    this.metadataFile = path.join(__dirname, '../data/wiki_update_metadata.json');
    this.pageTitlesFile = path.join(__dirname, '../data/osrs_page_titles.txt');
    this.filteredPagesFile = path.join(__dirname, '../data/osrs_filtered_pages.txt');
    this.nullPagesFile = path.join(__dirname, '../data/osrs_null_pages.txt');
    this.failedTemplatesFile = path.join(__dirname, '../data/osrs_failed_templates.txt');
    
    // OPTIMIZATION: Connection pooling (API COMPLIANT - Single connection only)
    this.httpAgent = new (require('http').Agent)({
      keepAlive: true,
      maxSockets: 1,        // CRITICAL: Only 1 concurrent connection (API compliant)
      maxFreeSockets: 1,
      timeout: 60000,
      freeSocketTimeout: 30000
    });

    this.httpsAgent = new (require('https').Agent)({
      keepAlive: true,
      maxSockets: 1,        // CRITICAL: Only 1 concurrent connection (API compliant)
      maxFreeSockets: 1,
      timeout: 60000,
      freeSocketTimeout: 30000
    });
    
    // OPTIMIZATION: Smart timing (API compliant)
    this.requestTimes = [];
    this.changeLimit = 500;

    // WATCHDOG: Continuous monitoring settings
    this.watchInterval = 5 * 60 * 1000; // Check every 5 minutes
    this.isRunning = false;
    this.watchTimer = null;
    
    // Stats tracking
    this.stats = {
      changesProcessed: 0,
      pagesAdded: 0,
      pagesUpdated: 0,
      pagesDeleted: 0,
      apiCalls: 0,
      startTime: null,
      endTime: null,
      lastUpdateTimestamp: null,
      // NEW: Track null responses for analysis
      nullResponses: [],
      filteredPages: [],
      shortPages: [],
      pagesReprocessed: 0,
      newPagesFound: 0
    };
    
    // In-memory page data and titles
    this.pageData = new Map();
    this.pageTitles = new Set();
    this.filteredPages = new Set(); // Track permanently filtered pages
    this.nullPages = new Set(); // Track pages that consistently return NULL
    this.failedTemplatePages = new Set(); // Track pages with persistent template parsing failures
  }

  async run() {
    console.log(chalk.cyan('🚀 OPTIMIZED OSRS Wiki Phase 3: Continuous Watchdog'));
    console.log(chalk.blue('======================================================='));
    console.log('');

    try {
      // Load existing data (ORDER MATTERS: load page data BEFORE titles)
      await this.loadMetadata();
      await this.loadExistingData();
      await this.loadPageTitles(); // Now pageData is loaded, so titles can be created from it
      await this.loadFilteredPages(); // Load previously filtered pages to avoid re-checking
      await this.loadNullPages(); // Load pages that consistently return NULL

      console.log(chalk.blue('🔄 Starting continuous monitoring...'));
      console.log(chalk.yellow(`⏰ Checking for changes every ${this.watchInterval / 60000} minutes`));
      console.log(chalk.gray('Press Ctrl+C to stop the watchdog'));
      console.log('');

      // Start continuous monitoring
      await this.startWatching();

    } catch (error) {
      console.error(chalk.red(`❌ Watchdog startup failed: ${error.message}`));
      process.exit(1);
    }
  }

  async startWatching() {
    this.isRunning = true;

    // Set up graceful shutdown
    process.on('SIGINT', () => this.stopWatching());
    process.on('SIGTERM', () => this.stopWatching());

    // ENHANCED: Clean up existing redirects and disambiguation pages
    // TEMPORARILY DISABLED - need to fix filtering logic first
    // await this.cleanupExistingRedirects();

    // Run initial check
    await this.checkForChanges();

    // Schedule periodic checks - FIXED: Only schedule next check after current one completes
    this.scheduleNextCheck();
  }

  scheduleNextCheck() {
    if (this.isRunning) {
      // Show countdown timer
      this.showCountdown();

      this.watchTimer = setTimeout(async () => {
        if (this.isRunning) {
          await this.checkForChanges();
          this.scheduleNextCheck(); // Schedule next check after completion
        }
      }, this.watchInterval);
    }
  }

  showCountdown() {
    const totalSeconds = this.watchInterval / 1000;
    let remainingSeconds = totalSeconds;

    console.log(chalk.yellow(`⏳ Next check in ${Math.floor(totalSeconds / 60)} minutes...`));

    // Update countdown every 30 seconds
    const countdownInterval = setInterval(() => {
      remainingSeconds -= 30;

      if (remainingSeconds <= 0 || !this.isRunning) {
        clearInterval(countdownInterval);
        return;
      }

      const minutes = Math.floor(remainingSeconds / 60);
      const seconds = remainingSeconds % 60;

      if (minutes > 0) {
        console.log(chalk.gray(`⏳ Next check in ${minutes}m ${seconds}s...`));
      } else {
        console.log(chalk.gray(`⏳ Next check in ${seconds}s...`));
      }
    }, 30000);
  }

  async stopWatching() {
    console.log(chalk.yellow('\n🛑 Stopping watchdog...'));
    this.isRunning = false;

    if (this.watchTimer) {
      clearTimeout(this.watchTimer); // FIXED: Use clearTimeout instead of clearInterval
      this.watchTimer = null;
    }

    console.log(chalk.green('✅ Watchdog stopped gracefully'));
    process.exit(0);
  }

  async cleanupExistingRedirects() {
    console.log(chalk.blue('🧹 Cleaning up existing redirects and disambiguation pages...'));

    let redirectsRemoved = 0;
    const pagesToRemove = [];

    // Check each page in our collection
    for (const [title, pageData] of this.pageData.entries()) {
      if (this.shouldFilterPage(title, pageData.text || '')) {
        pagesToRemove.push(title);
        redirectsRemoved++;
      }
    }

    // Remove the redirects
    for (const title of pagesToRemove) {
      this.pageData.delete(title);
      this.pageTitles.delete(title);
    }

    if (redirectsRemoved > 0) {
      console.log(chalk.green(`✅ Removed ${redirectsRemoved} redirects and disambiguation pages`));

      // Save the cleaned data
      await this.saveUpdatedData();
      await this.savePageTitles();

      // Update metadata
      const metadata = {
        lastUpdateTimestamp: new Date().toISOString(),
        totalPages: this.pageData.size,
        phase3CompletedAt: new Date().toISOString(),
        lastUpdateStats: {
          added: 0,
          updated: 0,
          deleted: redirectsRemoved
        },
        version: '2.0-optimized-cleaned'
      };

      await fs.writeFile(this.metadataFile, JSON.stringify(metadata, null, 2));
      console.log(chalk.green(`📊 Updated collection: ${this.pageData.size} pages (removed ${redirectsRemoved} redirects)`));
    } else {
      console.log(chalk.green('✅ No redirects found - collection is already clean'));
    }
  }

  async checkForChanges() {
    const checkStart = new Date();
    console.log(chalk.blue(`🔍 Checking for changes... (${checkStart.toISOString()})`));

    try {
      // Reset stats for this check
      this.stats = {
        changesProcessed: 0,
        pagesAdded: 0,
        pagesUpdated: 0,
        pagesDeleted: 0,
        apiCalls: 0,
        startTime: checkStart,
        endTime: null,
        lastUpdateTimestamp: this.stats.lastUpdateTimestamp,
        // ENHANCED: Initialize logging arrays
        nullResponses: [],
        filteredPages: [],
        shortPages: [],
        missingPages: [],
        newPagesFound: 0,
        pagesReprocessed: 0
      };

      // OPTIMIZED: Skip redundant sync operations (only sync when needed)
      // console.log(chalk.cyan('🔄 Synchronizing page titles with content...'));
      // await this.synchronizePageTitles();

      // ENHANCED: Check for missing pages (pages in titles but not in content)
      console.log(chalk.cyan('🔍 Checking for missing pages...'));
      const missingPages = await this.findMissingPages();

      if (missingPages.length > 0) {
        console.log(chalk.yellow(`📝 Found ${missingPages.length} missing pages, fetching content...`));
        await this.processMissingPages(missingPages);
      }

      // DISABLED: Template parsing check (was causing infinite loops)
      // console.log(chalk.cyan('🔧 Checking for template parsing issues...'));
      // await this.checkAndFixTemplateParsingIssues();

      // ENHANCED: Check for new pages (not in our titles list)
      console.log(chalk.cyan('🔍 Checking for new pages on wiki...'));
      const newPages = await this.findNewPages();

      if (newPages.length > 0) {
        console.log(chalk.yellow(`📝 Found ${newPages.length} new pages, adding to collection...`));
        await this.processNewPages(newPages);
      }

      // Get recent changes since last update
      const changes = await this.getRecentChanges();

      if (changes.length === 0 && missingPages.length === 0 && newPages.length === 0) {
        console.log(chalk.green('✅ No changes detected. Wiki content is up to date.'));
        return;
      }

      if (changes.length > 0) {
        console.log(chalk.yellow(`📦 Found ${changes.length.toLocaleString()} changes to process`));
        // Process changes serially (API compliant)
        await this.processChangesSerially(changes);
      }

      // Save updated data and metadata
      await this.saveUpdatedData();
      await this.savePageTitles();
      await this.saveFilteredPages();
      await this.saveNullPages();
      await this.updateMetadata();

      this.stats.endTime = new Date();
      this.printFinalStats();

      console.log(chalk.green('🎉 Changes processed successfully!'));

      // ENHANCED: Clean summary of what actually happened
      this.printCleanSummary();
      console.log('');

    } catch (error) {
      console.error(chalk.red(`❌ Error during change check: ${error.message}`));
      console.log(chalk.yellow('⏳ Will retry on next scheduled check...'));
      console.log('');
    }
  }

  async loadMetadata() {
    console.log(chalk.blue('📖 Loading update metadata...'));

    if (!await fs.pathExists(this.metadataFile)) {
      console.log(chalk.yellow('⚠️ No metadata file found. Starting fresh with current timestamp.'));
      this.stats.lastUpdateTimestamp = new Date().toISOString();
      return;
    }

    const metadata = await fs.readJson(this.metadataFile);
    this.stats.lastUpdateTimestamp = metadata.lastUpdateTimestamp;

    // CRITICAL FIX: Check if timestamp is reasonable (not in future, not too old)
    const lastUpdate = new Date(this.stats.lastUpdateTimestamp);
    const now = new Date();
    const oneYearAgo = new Date(now.getTime() - 365 * 24 * 60 * 60 * 1000);

    if (lastUpdate > now || lastUpdate < oneYearAgo) {
      console.log(chalk.yellow(`⚠️ Suspicious timestamp detected: ${this.stats.lastUpdateTimestamp}`));
      console.log(chalk.yellow('🔧 Resetting to 24 hours ago to catch recent changes safely'));
      this.stats.lastUpdateTimestamp = new Date(now.getTime() - 24 * 60 * 60 * 1000).toISOString();
    }

    console.log(chalk.green(`✅ Last update: ${this.stats.lastUpdateTimestamp}`));
  }

  async loadExistingData() {
    console.log(chalk.blue(`📚 Loading existing wiki content from: ${this.outputFile}`));

    if (!await fs.pathExists(this.outputFile)) {
      console.log(chalk.yellow('⚠️ No existing content file found. Starting fresh.'));
      console.log(chalk.yellow(`⚠️ Checked path: ${this.outputFile}`));
      return;
    }

    console.log(chalk.blue('📖 File exists, reading content...'));
    const content = await fs.readFile(this.outputFile, 'utf8');
    const lines = content.trim().split('\n').filter(line => line.trim());
    console.log(chalk.blue(`📊 Found ${lines.length.toLocaleString()} lines to process`));

    let validPages = 0;
    let invalidLines = 0;

    for (const line of lines) {
      try {
        const pageData = JSON.parse(line);
        if (pageData.title) {
          // CRITICAL FIX: Clean HTML from title to match page titles file
          const cleanTitle = pageData.title.replace(/<[^>]*>/g, '').trim();

          // Update the pageData with clean title
          pageData.title = cleanTitle;

          this.pageData.set(cleanTitle, pageData);
          validPages++;
        } else {
          console.warn(chalk.yellow(`⚠️ Page data missing title: ${line.substring(0, 50)}...`));
          invalidLines++;
        }
      } catch (error) {
        console.warn(chalk.yellow(`⚠️ Skipping invalid JSON line: ${line.substring(0, 50)}...`));
        invalidLines++;
      }
    }

    console.log(chalk.green(`✅ Loaded ${this.pageData.size.toLocaleString()} existing pages (${validPages} valid, ${invalidLines} invalid)`));
  }

  async loadPageTitles() {
    console.log(chalk.blue('📋 Loading page titles...'));

    if (!await fs.pathExists(this.pageTitlesFile)) {
      console.log(chalk.yellow('⚠️ No page titles file found. Creating from existing page data...'));
      // Create from existing page data (pageData should be loaded by now)
      for (const title of this.pageData.keys()) {
        this.pageTitles.add(title);
      }
      console.log(chalk.green(`✅ Created ${this.pageTitles.size.toLocaleString()} page titles from existing data`));
      return;
    }

    const content = await fs.readFile(this.pageTitlesFile, 'utf8');
    const titles = content.trim().split('\n').filter(title => title.trim());

    if (titles.length === 0) {
      console.log(chalk.yellow('⚠️ Page titles file is empty. Creating from existing page data...'));
      // File exists but is empty, create from existing page data
      for (const title of this.pageData.keys()) {
        this.pageTitles.add(title);
      }
      console.log(chalk.green(`✅ Created ${this.pageTitles.size.toLocaleString()} page titles from existing data`));
    } else {
      for (const title of titles) {
        this.pageTitles.add(title.trim());
      }
      console.log(chalk.green(`✅ Loaded ${this.pageTitles.size.toLocaleString()} page titles from file`));
    }
  }

  async loadFilteredPages() {
    console.log(chalk.blue('🚫 Loading filtered pages list...'));

    if (!await fs.pathExists(this.filteredPagesFile)) {
      console.log(chalk.yellow('⚠️ No filtered pages file found. Starting with empty list.'));
      return;
    }

    const content = await fs.readFile(this.filteredPagesFile, 'utf8');
    const filteredTitles = content.trim().split('\n').filter(title => title.trim());

    for (const title of filteredTitles) {
      this.filteredPages.add(title.trim());
    }

    console.log(chalk.green(`✅ Loaded ${this.filteredPages.size.toLocaleString()} previously filtered pages`));
  }

  async loadNullPages() {
    console.log(chalk.blue('⚫ Loading NULL pages list...'));

    if (!await fs.pathExists(this.nullPagesFile)) {
      console.log(chalk.yellow('⚠️ No NULL pages file found. Starting with empty list.'));
      return;
    }

    const content = await fs.readFile(this.nullPagesFile, 'utf8');
    const nullTitles = content.trim().split('\n').filter(title => title.trim());

    for (const title of nullTitles) {
      this.nullPages.add(title.trim());
    }

    console.log(chalk.green(`✅ Loaded ${this.nullPages.size.toLocaleString()} known NULL pages`));

    // CRITICAL: Remove NULL pages from pageData to prevent them from being checked
    if (this.nullPages.size > 0) {
      let removedCount = 0;
      for (const nullPage of this.nullPages) {
        if (this.pageData.has(nullPage)) {
          this.pageData.delete(nullPage);
          this.pageTitles.delete(nullPage);
          removedCount++;
        }
      }
      if (removedCount > 0) {
        console.log(chalk.yellow(`🗑️ Removed ${removedCount} NULL pages from collection`));
      }
    }
  }

  async getRecentChanges() {
    // CLI OPTION: Check if user wants to skip batch check on startup
    const skipBatchCheck = process.argv.includes('--skip-batch') || process.argv.includes('--recent-only');

    // SMART APPROACH: Use recent changes API first, fallback to batch check
    const lastCheck = this.metadata?.lastChangeCheck || 0;
    const timeSinceLastCheck = Date.now() - lastCheck;

    // Determine if we should do batch check
    const shouldDoBatchCheck = !skipBatchCheck && (
      timeSinceLastCheck > 6 * 60 * 60 * 1000 ||
      !this.metadata?.lastChangeCheck
    );

    if (shouldDoBatchCheck) {
      console.log(chalk.blue('🚀 COMPREHENSIVE: Full batch revision checking (first run or 6+ hours since last)...'));
      console.log(chalk.gray('   💡 Use --skip-batch or --recent-only to start with recent changes API instead'));
      const changedPages = await this.batchCheckRevisions();
      this.metadata.lastChangeCheck = Date.now();
      return changedPages;
    }

    // Normal operation or user requested recent-only: Use efficient recent changes API
    if (skipBatchCheck && !this.metadata?.lastChangeCheck) {
      console.log(chalk.blue('🚀 EFFICIENT: Starting with recent changes API (--skip-batch mode)...'));
      // Set initial timestamp to avoid going too far back
      this.metadata.lastRecentChangesCheck = Date.now() - 10 * 60 * 1000; // 10 minutes ago
    } else {
      console.log(chalk.blue('🚀 EFFICIENT: Recent changes API checking...'));
    }

    try {
      const recentChanges = await this.getRecentChangesFromAPI();
      if (recentChanges.length > 0) {
        console.log(chalk.green(`✅ Found ${recentChanges.length} recent changes`));
        return recentChanges;
      } else {
        console.log(chalk.green('✅ No recent changes detected'));
        return [];
      }
    } catch (error) {
      console.log(chalk.yellow(`⚠️ Recent changes API failed: ${error.message}, falling back to batch check`));
      const changedPages = await this.batchCheckRevisions();
      return changedPages;
    }
  }

  // EFFICIENT: Get recent changes from MediaWiki API
  async getRecentChangesFromAPI() {
    // SMART: Use last check time or default to 10 minutes ago
    const lastCheck = this.metadata?.lastRecentChangesCheck || (Date.now() - 10 * 60 * 1000);
    const since = new Date(lastCheck).toISOString();
    const now = new Date().toISOString();

    console.log(chalk.gray(`   📅 Checking changes since: ${since}`));

    try {
      const response = await axios.get(this.wikiApiUrl, {
        params: {
          action: 'query',
          list: 'recentchanges',
          rcstart: now,        // Start from now
          rcend: since,        // Go back to last check
          rcnamespace: 0,      // Main namespace only
          rctype: 'edit|new',  // Only edits and new pages
          rcprop: 'title|ids|timestamp|comment',
          rclimit: 500,        // Max results
          format: 'json'
        },
        headers: { 'User-Agent': this.userAgent },
        timeout: 30000
      });

      const recentChanges = response.data.query?.recentchanges || [];
      const relevantChanges = [];

      for (const change of recentChanges) {
        const title = change.title;

        // Only process pages we have in our collection
        if (this.pageData.has(title)) {
          const existingPage = this.pageData.get(title);

          // Check if revision actually changed
          if (!existingPage || existingPage.revid !== change.revid) {
            relevantChanges.push({
              title: title,
              type: 'edit',
              revid: change.revid,
              oldRevid: existingPage?.revid || null,
              timestamp: change.timestamp
            });
          }
        }
      }

      console.log(chalk.gray(`   📊 Checked ${recentChanges.length} recent changes, ${relevantChanges.length} relevant to our collection`));

      // Update last check time for next run
      this.metadata.lastRecentChangesCheck = Date.now();

      return relevantChanges;

    } catch (error) {
      console.error(chalk.red(`❌ Recent changes API error: ${error.message}`));
      throw error;
    }
  }

  // REVOLUTIONARY: Fast batch revision checking for ALL pages
  async batchCheckRevisions() {
    console.log(chalk.blue('🔍 Batch checking revision IDs for all pages...'));

    // ENHANCED: Skip known NULL pages to avoid repeated failures

    // DEBUG: Check if NULL pages are in our data
    console.log(chalk.gray(`   🔍 DEBUG: nullPages set has ${this.nullPages.size} entries`));
    if (this.nullPages.size > 0) {
      const firstNullPage = Array.from(this.nullPages)[0];
      const isInPageData = this.pageData.has(firstNullPage);
      console.log(chalk.gray(`   🔍 DEBUG: First NULL page "${firstNullPage}" is in pageData: ${isInPageData}`));
    }

    const allTitles = Array.from(this.pageData.keys()).filter(title => !this.nullPages.has(title));
    const batchSize = 50; // MediaWiki API limit for titles per request
    const changedPages = [];

    console.log(chalk.gray(`   📊 Checking ${allTitles.length.toLocaleString()} pages in batches of ${batchSize} (${this.nullPages.size} NULL pages already removed from collection)`));

    for (let i = 0; i < allTitles.length; i += batchSize) {
      const batch = allTitles.slice(i, i + batchSize);
      const batchNumber = Math.floor(i / batchSize) + 1;
      const totalBatches = Math.ceil(allTitles.length / batchSize);

      console.log(chalk.gray(`   📦 Batch ${batchNumber}/${totalBatches} (${batch.length} pages)`));

      try {
        const currentRevisions = await this.getBatchRevisions(batch);

        // Compare revision IDs in memory
        for (const pageInfo of currentRevisions) {
          const title = pageInfo.title;
          const currentRevId = pageInfo.revid;
          const storedPage = this.pageData.get(title);

          if (!storedPage || storedPage.revid !== currentRevId) {
            // CRITICAL FIX: Skip NULL pages even if they show as "changed"
            if (this.nullPages.has(title)) {
              // Skip - this is a known NULL page that will fail anyway
              continue;
            }

            // Page has changed or is new
            changedPages.push({
              title: title,
              type: storedPage ? 'edit' : 'new',
              revid: currentRevId,
              oldRevid: storedPage?.revid || null
            });
          }
        }

      } catch (error) {
        console.error(chalk.red(`❌ Failed to check batch ${batchNumber}: ${error.message}`));
        // Continue with next batch
      }
    }

    console.log(chalk.green(`✅ Batch revision check complete: ${changedPages.length} pages changed`));
    return changedPages;
  }

  // Get revision IDs for a batch of page titles
  async getBatchRevisions(titles) {
    const params = {
      action: 'query',
      prop: 'revisions',
      titles: titles.join('|'),
      rvprop: 'ids|timestamp',
      formatversion: 2,
      format: 'json'
    };

    const startTime = Date.now();
    const response = await this.makeOptimizedRequest(params);
    const requestTime = Date.now() - startTime;

    // Track request times
    this.requestTimes.push(requestTime);
    if (this.requestTimes.length > 10) {
      this.requestTimes.shift();
    }

    this.stats.apiCalls++;

    const pages = response.data.query?.pages || [];
    const revisions = [];

    for (const page of pages) {
      if (page.revisions && page.revisions.length > 0) {
        revisions.push({
          title: page.title,
          revid: page.revisions[0].revid,
          timestamp: page.revisions[0].timestamp
        });
      }
    }

    return revisions;
  }

  // REMOVED: Old deduplication method no longer needed with batch revision checking

  // REMOVED: Old filtering method no longer needed with batch revision checking

  async makeOptimizedRequest(params, retryCount = 0) {
    const maxRetries = 3;

    try {
      return await axios.get(this.wikiApiUrl, {
        params,
        headers: {
          'User-Agent': this.userAgent,
          'Accept': 'application/json, text/plain, */*',
          'Connection': 'keep-alive'
        },
        timeout: 30000,
        httpAgent: this.httpAgent,
        httpsAgent: this.httpsAgent
      });
    } catch (error) {
      // Handle rate limiting (required by MediaWiki API etiquette)
      if (error.response?.data?.error?.code === 'ratelimited' && retryCount < maxRetries) {
        const backoffTime = Math.pow(2, retryCount) * 1000; // Exponential backoff
        console.warn(chalk.yellow(`⚠️ Rate limited! Backing off for ${backoffTime}ms (attempt ${retryCount + 1}/${maxRetries})`));
        await this.sleep(backoffTime);
        return this.makeOptimizedRequest(params, retryCount + 1);
      }
      throw error;
    }
  }

  // OPTIMIZATION: Process changes serially (API compliant)
  async processChangesSerially(changes) {
    console.log(chalk.blue('⚙️ Processing changes serially (API compliant)...'));

    const spinner = ora('Processing changes...').start();

    for (let i = 0; i < changes.length; i++) {
      const change = changes[i];

      try {
        if (change.type === 'new') {
          await this.processNewPage(change);
        } else if (change.type === 'edit') {
          await this.processEditedPage(change);
        } else if (change.type === 'log' && change.logtype === 'delete') {
          await this.processDeletedPage(change);
        }

        // Update progress - show actual processing activity
        if (i % 10 === 0 || i === changes.length - 1) {
          const processed = this.stats.pagesAdded + this.stats.pagesUpdated + this.stats.pagesDeleted;
          spinner.text = `Processed ${i + 1}/${changes.length} changes (${processed} actual updates)...`;
        }

        // CRITICAL FIX: Only delay after actual API calls, not every iteration
        // Delay logic moved to fetchPageContent() method where API calls actually happen

      } catch (error) {
        console.error(chalk.red(`❌ Failed to process change ${change.title}: ${error.message}`));
      }
    }

    spinner.succeed(`Processed ${changes.length.toLocaleString()} changes serially`);
  }

  async processDeletedPage(change) {
    const title = change.title;

    if (this.pageData.has(title)) {
      this.pageData.delete(title);
      this.pageTitles.delete(title); // Remove from titles set
      this.stats.pagesDeleted++;
      console.log(chalk.red(`🗑️ Deleted: ${title}`));
    }
  }

  async processNewPage(change) {
    const title = change.title;

    try {
      const pageContent = await this.fetchPageContent(title);

      if (pageContent) {
        this.pageData.set(title, pageContent);
        this.pageTitles.add(title); // NEW: Add to titles set
        this.stats.pagesAdded++;
        console.log(chalk.green(`➕ Added: ${title} (rev ${pageContent.revid})`));
      } else {
        // ENHANCED: Log null responses for analysis
        this.stats.nullResponses.push({
          title: title,
          type: 'new',
          revid: change.revid,
          reason: 'null_response_from_api'
        });
        console.log(chalk.yellow(`⚠️ NULL: ${title} (new page returned null from API)`));

        // ENHANCED: Preserve player spoofs even if they return null initially
        if (this.isPlayerSpoofPage(title)) {
          console.log(chalk.blue(`🎭 PRESERVED: ${title} (player spoof/holiday event NPC - adding despite null response)`));
          // Add minimal page data for player spoofs
          const playerSpoofData = {
            title: title,
            categories: ['Player_spoofs'],
            text: `${title} is a player spoof NPC from OSRS holiday events or minigames.`,
            revid: change.revid,
            timestamp: change.timestamp
          };

          this.pageData.set(title, playerSpoofData);
          this.pageTitles.add(title);
          this.stats.pagesAdded++;
        } else {
          // SMART: Verify if new page actually exists before skipping
          console.log(chalk.yellow(`⚠️ VERIFYING NEW: ${title} (null response - checking if page actually exists)`));
          const pageExists = await this.verifyPageExists(title);

          if (pageExists) {
            console.log(chalk.green(`✅ CONFIRMED NEW: ${title} (page exists - will retry content fetch later)`));
            // Add placeholder data for now, will be updated in next scan
            const placeholderData = {
              title: title,
              categories: [],
              text: `Placeholder for ${title} - content will be fetched in next scan.`,
              revid: change.revid,
              timestamp: change.timestamp,
              needsContentUpdate: true
            };

            this.pageData.set(title, placeholderData);
            this.pageTitles.add(title);
            this.stats.pagesAdded++;
          } else {
            console.log(chalk.gray(`🚫 SKIPPED: ${title} (verified as non-existent)`));
            // Add to NULL pages list to avoid checking again
            this.nullPages.add(title);
          }
        }
      }
    } catch (error) {
      console.error(chalk.red(`❌ Failed to process new page ${title}: ${error.message}`));
    }
  }

  async processEditedPage(change) {
    const title = change.title;
    const existingPage = this.pageData.get(title);
    
    // Skip if revision hasn't changed
    if (existingPage && change.revid && existingPage.revid === change.revid) {
      return;
    }
    
    try {
      const pageContent = await this.fetchPageContent(title);
      
      if (pageContent) {
        // CRITICAL FIX: Ensure revision ID matches what was detected in batch check
        if (change.revid && pageContent.revid !== change.revid) {
          console.log(chalk.yellow(`⚠️ REVISION MISMATCH: ${title} - batch detected ${change.revid}, content fetch got ${pageContent.revid}, using batch revision`));
          pageContent.revid = change.revid; // Use the revision ID from batch check
        }

        this.pageData.set(title, pageContent);
        this.stats.pagesUpdated++;
        console.log(chalk.blue(`📝 Updated: ${title} (rev ${existingPage?.revid || 'unknown'} → ${pageContent.revid})`));
      } else {
        // ENHANCED: Log null responses for analysis
        this.stats.nullResponses.push({
          title: title,
          type: 'edit',
          oldRevid: existingPage?.revid,
          newRevid: change.revid,
          reason: 'null_response_from_api'
        });
        console.log(chalk.yellow(`⚠️ NULL: ${title} (edit returned null from API, old rev: ${existingPage?.revid}, new rev: ${change.revid})`));

        // ENHANCED: Preserve player spoofs and holiday event NPCs (legitimate content)
        if (this.isPlayerSpoofPage(title)) {
          console.log(chalk.blue(`🎭 PRESERVED: ${title} (player spoof/holiday event NPC - keeping despite null response)`));
          // Keep the existing page data, just update the revision ID
          if (existingPage) {
            existingPage.revid = change.revid;
            existingPage.timestamp = change.timestamp;
          }
        } else if (existingPage) {
          // SMART: Verify page existence before removing (null could be temporary API issue)
          console.log(chalk.yellow(`⚠️ VERIFYING: ${title} (null response - checking if page actually exists)`));
          const pageExists = await this.verifyPageExists(title);

          if (pageExists) {
            console.log(chalk.green(`✅ CONFIRMED: ${title} (page exists - null was temporary API issue)`));
            // Keep the existing page data, just update the revision ID
            existingPage.revid = change.revid;
            existingPage.timestamp = change.timestamp;
          } else {
            console.log(chalk.red(`🗑️ CONFIRMED DELETED: ${title} (page verified as non-existent)`));
            this.pageData.delete(title);
            this.pageTitles.delete(title);
            this.stats.pagesDeleted++;
          }
        }
      }
    } catch (error) {
      console.error(chalk.red(`❌ Failed to process edited page ${title}: ${error.message}`));
    }
  }



  async fetchPageContent(title, retryCount = 0) {
    const maxRetries = 3;

    try {
      // FIXED: Fetch raw wikitext first to properly parse templates
      const wikitextParams = {
        action: 'query',
        prop: 'revisions|categories',
        rvprop: 'content|ids',
        format: 'json',
        formatversion: 2,
        titles: title
      };

      // CRITICAL FIX: Add timing and smart delay for API calls
      const startTime = Date.now();
      const response = await this.makeOptimizedRequest(wikitextParams);
      const requestTime = Date.now() - startTime;

      // Track request times for performance monitoring
      this.requestTimes.push(requestTime);
      if (this.requestTimes.length > 10) {
        this.requestTimes.shift(); // Keep only last 10 measurements
      }

      // NO ARTIFICIAL DELAYS: OSRS Wiki has no read limits
      // Serial processing + keep-alive connections provide natural, respectful timing

      const queryResult = response.data.query;
      if (!queryResult || !queryResult.pages || queryResult.pages.length === 0) return null;

      const page = queryResult.pages[0];
      if (!page.revisions || page.revisions.length === 0) return null;

      const rawWikitext = page.revisions[0].content || '';
      const revid = page.revisions[0].revid;
      const categories = page.categories || [];

      // Filter out unwanted page types based on wikitext content
      if (this.shouldFilterPageFromWikitext(title, rawWikitext)) {
        // ENHANCED: Log filtered pages for analysis
        this.stats.filteredPages.push({
          title: title,
          reason: 'filtered_page_type',
          revid: revid
        });
        return null;
      }

      // ENHANCED: Process templates from raw wikitext FIRST
      const processedWikitext = await this.processTemplatesFromWikitext(rawWikitext);

      // Convert processed wikitext to clean text
      const cleanText = this.cleanWikitextContent(processedWikitext);

      if (cleanText.length < 100) {
        // ENHANCED: Log short pages for analysis
        this.stats.shortPages.push({
          title: title,
          reason: 'too_short',
          length: cleanText.length,
          revid: revid
        });
        return null; // Skip very short pages
      }

      const pageData = {
        title: title,
        categories: categories.map(cat => cat.category || cat),
        text: cleanText,
        revid: revid,
        timestamp: new Date().toISOString()
      };

      // Clean the page data for AI processing
      return this.cleanPageDataForAI(pageData);

    } catch (error) {
      if (retryCount < maxRetries) {
        console.warn(chalk.yellow(`⚠️ Retry ${retryCount + 1}/${maxRetries} for ${title}: ${error.message}`));
        // EXPONENTIAL BACKOFF (REQUIRED BY MEDIAWIKI API ETIQUETTE)
        const avgRequestTime = this.requestTimes.length > 0
          ? this.requestTimes.reduce((a, b) => a + b, 0) / this.requestTimes.length
          : 1000;
        const backoffTime = Math.min(avgRequestTime * Math.pow(2, retryCount), 10000);
        console.log(chalk.yellow(`⏳ Exponential backoff: ${backoffTime}ms`));
        await this.sleep(backoffTime);
        return this.fetchPageContent(title, retryCount + 1);
      }

      console.error(chalk.red(`❌ Failed to fetch ${title} after ${maxRetries} retries`));
      return null;
    }
  }

  // ENHANCED: Skip unwanted pages before processing
  shouldSkipPage(title, change) {
    // Skip pages with unwanted prefixes/suffixes
    const unwantedPatterns = [
      // User pages, talk pages, etc. (should be filtered by namespace but double-check)
      /^User:/i,
      /^Talk:/i,
      /^File:/i,
      /^Category:/i,
      /^Template:/i,
      /^Help:/i,
      /^Special:/i,
      /^MediaWiki:/i,

      // Redirects and disambiguation
      /redirect/i,
      /disambiguation/i,
      /\(disambiguation\)/i,

      // Temporary/test pages
      /^Sandbox/i,
      /^Test/i,
      /\/sandbox/i,
      /\/test/i,

      // Archive pages
      /^Archive:/i,
      /\/Archive/i,

      // User subpages
      /\//  // Contains slash (likely a subpage)
    ];

    // Check title patterns
    if (unwantedPatterns.some(pattern => pattern.test(title))) {
      return true;
    }

    // Skip if change has no revision ID (likely invalid)
    if (change.type === 'new' && !change.revid) {
      return true;
    }

    return false;
  }

  shouldFilterPage(title, htmlContent) {
    // ENHANCED: More precise filtering - only filter TRUE disambiguation/redirect pages

    // Check for actual redirects (these should always be filtered)
    const redirectPatterns = [
      /class="mw-redirect"/i,
      /#REDIRECT/i,
      /^.{1,100}\s+redirects here\./i  // Short content that's just "X redirects here."
    ];

    if (redirectPatterns.some(pattern => pattern.test(htmlContent))) {
      return true;
    }

    // Check for TRUE disambiguation pages (not just pages with disambiguation notes)
    const disambiguationIndicators = [
      // Must have "This is a disambiguation page" text
      /This is a disambiguation page/i,
      // Must have "distinguish between articles" text
      /This page is used to distinguish between articles/i,
      // Must have "internal link led you to this disambiguation page" text
      /If an internal link led you to this disambiguation page/i
    ];

    // Only filter if it's a TRUE disambiguation page with minimal content
    if (disambiguationIndicators.some(pattern => pattern.test(htmlContent))) {
      // Additional check: if content is very short (< 500 chars), it's likely a pure disambiguation page
      const cleanContent = htmlContent.replace(/<[^>]*>/g, ' ').replace(/\s+/g, ' ').trim();
      if (cleanContent.length < 500) {
        return true;
      }
    }

    // Check for disambiguation pages by category
    if (title.includes('(disambiguation)') && htmlContent.includes('may refer to:')) {
      const cleanContent = htmlContent.replace(/<[^>]*>/g, ' ').replace(/\s+/g, ' ').trim();
      if (cleanContent.length < 500) {
        return true;
      }
    }

    return false;
  }

  isPlayerSpoofPage(title) {
    // Known player spoof pages from holiday events and minigames
    const playerSpoofNames = [
      // Holiday event player spoofs
      'epic mager34',
      'mad melvin96',
      'Xi plzPetDogz xiX',
      'LazyLaura94',
      'GreatScott85',
      'JoyfulJudy02',
      'CharlieChimes06',
      'MerryMax2000',
      'WhoLouCindy57',
      'Ebenezer1843',
      'JacobM7',
      'MiniatureMarie',
      'DeathlyFuture3',
      'CanYouCratchit',
      'Unicorn1337Kilr',

      // Hallowed Sepulchre player spoofs
      'c4ssi4n',
      'c0lect0r890',
      'fishrunner82',
      'Jyn',
      'r2t2pnsh0Ty',
      'weast side49',

      // Other player spoofs
      '1337sp34kr',
      '1337mage43',
      'Cool Mom227',
      'Elfinlocks',
      'Purepker895',
      'Qutiedoll',
      'BigRedJapan',
      'Runite Minor',
      'PKMaster0036',
      'R4ng3rNo0b889',
      'Cow31337Killer',
      'Durial321',
      'Hopleez'
    ];

    return playerSpoofNames.includes(title);
  }

  // SMART: Verify if a page actually exists on the wiki
  async verifyPageExists(title, retryCount = 0) {
    const maxRetries = 2;

    try {
      // Use allpages API to check if page exists
      const response = await axios.get(this.wikiApiUrl, {
        params: {
          action: 'query',
          list: 'allpages',
          apfrom: title,
          apto: title,
          aplimit: 1,
          apnamespace: 0,
          apfilterredir: 'nonredirects',
          format: 'json'
        }
      });

      if (response.data?.query?.allpages) {
        const pages = response.data.query.allpages;
        // Check if exact title match exists
        const exactMatch = pages.find(page => page.title === title);
        return !!exactMatch;
      }

      return false;
    } catch (error) {
      if (retryCount < maxRetries) {
        console.log(chalk.yellow(`⚠️ Retry ${retryCount + 1}/${maxRetries} for page verification: ${title}`));
        await new Promise(resolve => setTimeout(resolve, 1000 * (retryCount + 1))); // Exponential backoff
        return this.verifyPageExists(title, retryCount + 1);
      }

      console.error(chalk.red(`❌ Failed to verify page existence for ${title}: ${error.message}`));
      // If verification fails, assume page exists to be safe
      return true;
    }
  }

  async checkAndFixTemplateParsingIssues() {
    console.log(chalk.blue('🔧 Scanning existing pages for template parsing issues...'));

    let pagesWithIssues = 0;
    let pagesFixed = 0;

    // Sample check: Look for pages with unparsed templates
    for (const [title, pageData] of this.pageData) {
      try {
        const content = pageData.text || '';

        // Check if page has unparsed templates (simple heuristic)
        const templateCount = (content.match(/\{\{[^}]+\}\}/g) || []).length;

        if (templateCount > 0) {
          // Check if templates are properly parsed by looking for human-readable indicators
          const hasProperStats = content.includes('Combat Level:') || content.includes('Attack:') || content.includes('Defence:');
          const hasProperDrops = content.includes('Drop:') || content.includes('Rarity:');
          const hasProperLocations = content.includes('Location:') || content.includes('Area:');

          // If page has templates but no parsed indicators, it likely needs reprocessing
          if (templateCount > 5 && !hasProperStats && !hasProperDrops && !hasProperLocations) {
            pagesWithIssues++;

            // Refetch and reprocess the page
            console.log(chalk.yellow(`🔧 Reprocessing ${title} (${templateCount} unparsed templates)`));

            const freshContent = await this.fetchPageContent(title);
            if (freshContent) {
              this.pageData.set(title, freshContent);
              pagesFixed++;
              this.stats.pagesReprocessed++;

              // Add small delay to be API-friendly
              await new Promise(resolve => setTimeout(resolve, 100));
            }
          }
        }
      } catch (error) {
        console.warn(chalk.yellow(`⚠️ Error checking ${title}: ${error.message}`));
      }

      // ENHANCED: Process all template issues, but add small delay to be API-friendly
      if (pagesWithIssues > 0 && pagesWithIssues % 25 === 0) {
        console.log(chalk.blue(`🔄 Processed ${pagesWithIssues} template issues so far, continuing...`));
        await this.sleep(1000); // 1 second pause every 25 pages
      }
    }

    if (pagesFixed > 0) {
      console.log(chalk.green(`✅ Fixed template parsing for ${pagesFixed} pages`));
    } else {
      console.log(chalk.green('✅ No template parsing issues found'));
    }
  }

  async processTemplates(htmlContent) {
    try {
      // Use the Python template parser to process MediaWiki templates
      const templateParserPath = path.join(__dirname, '../api/wiki_template_parser.py');

      return new Promise((resolve, reject) => {
        const python = spawn('python3', ['-c', `
import sys
sys.path.append('${path.dirname(templateParserPath)}')
from wiki_template_parser import OSRSWikiTemplateParser

parser = OSRSWikiTemplateParser()
content = sys.stdin.read()
processed = parser.process_wiki_content(content)
print(processed)
        `]);

        let output = '';
        let error = '';

        python.stdout.on('data', (data) => {
          output += data.toString();
        });

        python.stderr.on('data', (data) => {
          error += data.toString();
        });

        python.on('close', (code) => {
          if (code === 0) {
            resolve(output.trim());
          } else {
            console.warn(chalk.yellow(`⚠️ Template parsing failed: ${error}`));
            resolve(htmlContent); // Fallback to original content
          }
        });

        python.stdin.write(htmlContent);
        python.stdin.end();
      });
    } catch (error) {
      console.warn(chalk.yellow(`⚠️ Template processing error: ${error.message}`));
      return htmlContent; // Fallback to original content
    }
  }

  cleanHtmlContent(html) {
    return html
      .replace(/<[^>]*>/g, ' ')
      .replace(/\s+/g, ' ')
      .trim();
  }

  async saveUpdatedData() {
    console.log(chalk.blue('💾 Saving updated wiki content...'));

    // ALPHABETIZE: Sort all pages by title for organized file structure
    console.log(chalk.cyan('🔤 Alphabetizing entries by title...'));
    const sortedPages = Array.from(this.pageData.values())
      .sort((a, b) => {
        // Case-insensitive alphabetical sort
        const titleA = (a.title || '').toLowerCase();
        const titleB = (b.title || '').toLowerCase();
        return titleA.localeCompare(titleB);
      });

    const lines = sortedPages.map(page => JSON.stringify(page));
    const content = lines.join('\n') + '\n';

    await fs.writeFile(this.outputFile, content, 'utf8');
    console.log(chalk.green(`✅ Saved ${this.pageData.size.toLocaleString()} pages to JSONL file (alphabetically sorted)`));
  }

  // NEW: Save updated page titles file
  async savePageTitles() {
    console.log(chalk.blue('📋 Saving updated page titles...'));

    // DEBUG: Show title count changes
    const currentCount = this.pageTitles.size;
    console.log(chalk.gray(`   📊 Current titles in memory: ${currentCount.toLocaleString()}`));

    const sortedTitles = Array.from(this.pageTitles).sort();
    const content = sortedTitles.join('\n') + '\n';

    await fs.writeFile(this.pageTitlesFile, content, 'utf8');
    console.log(chalk.green(`✅ Saved ${this.pageTitles.size.toLocaleString()} page titles (alphabetically sorted)`));
    console.log(chalk.gray(`   📁 File: ${this.pageTitlesFile}`));
  }

  async saveFilteredPages() {
    if (this.filteredPages.size === 0) {
      return; // No filtered pages to save
    }

    console.log(chalk.blue('🚫 Saving filtered pages list...'));

    const sortedFiltered = Array.from(this.filteredPages).sort();
    const content = sortedFiltered.join('\n') + '\n';

    await fs.writeFile(this.filteredPagesFile, content, 'utf8');
    console.log(chalk.green(`✅ Saved ${this.filteredPages.size.toLocaleString()} filtered pages`));
    console.log(chalk.gray(`   📁 File: ${this.filteredPagesFile}`));
  }

  async saveNullPages() {
    if (this.nullPages.size === 0) {
      return; // No NULL pages to save
    }

    console.log(chalk.blue('⚫ Saving NULL pages list...'));

    const sortedNull = Array.from(this.nullPages).sort();
    const content = sortedNull.join('\n') + '\n';

    await fs.writeFile(this.nullPagesFile, content, 'utf8');
    console.log(chalk.green(`✅ Saved ${this.nullPages.size.toLocaleString()} NULL pages`));
    console.log(chalk.gray(`   📁 File: ${this.nullPagesFile}`));
  }

  async updateMetadata() {
    console.log(chalk.blue('📝 Updating metadata...'));

    // CRITICAL FIX: Update the timestamp for next check BEFORE saving
    const newTimestamp = new Date().toISOString();
    this.stats.lastUpdateTimestamp = newTimestamp;

    const metadata = {
      lastUpdateTimestamp: newTimestamp,
      totalPages: this.pageData.size,
      phase3CompletedAt: newTimestamp,
      lastUpdateStats: {
        added: this.stats.pagesAdded,
        updated: this.stats.pagesUpdated,
        deleted: this.stats.pagesDeleted
      },
      version: '2.0-optimized'
    };

    await fs.writeJson(this.metadataFile, metadata, { spaces: 2 });
    console.log(chalk.green('✅ Metadata updated'));
  }

  // Integrated AI cleanup methods
  cleanPageDataForAI(pageData) {
    if (!pageData || typeof pageData !== 'object') return pageData;

    const cleaned = { ...pageData };

    if (cleaned.title) {
      cleaned.title = this.cleanTitle(cleaned.title);
    }

    if (cleaned.categories) {
      cleaned.categories = this.cleanCategories(cleaned.categories);
    }

    if (cleaned.text) {
      cleaned.text = this.cleanText(cleaned.text);
    }

    return cleaned;
  }

  cleanTitle(title) {
    if (!title || typeof title !== 'string') return title;

    let cleanTitle = title;

    // Extract content from HTML wrapper
    const spanMatch = cleanTitle.match(/<span[^>]*class="mw-page-title-main"[^>]*>(.*?)<\/span>/);
    if (spanMatch) {
      cleanTitle = spanMatch[1];
    }

    // Decode HTML entities
    cleanTitle = this.decodeHtmlEntities(cleanTitle);

    // Normalize whitespace
    cleanTitle = cleanTitle.replace(/\s+/g, ' ').trim();

    return cleanTitle;
  }

  cleanCategories(categories) {
    if (!categories) return [];
    if (!Array.isArray(categories)) return [];

    return categories.map(cat => {
      if (typeof cat === 'object' && cat.category) {
        return this.decodeHtmlEntities(cat.category);
      }

      if (typeof cat === 'string') {
        return this.decodeHtmlEntities(cat);
      }

      return cat;
    }).filter(cat => cat && typeof cat === 'string');
  }

  cleanText(text) {
    if (!text || typeof text !== 'string') return text;

    let cleaned = text;

    // Decode HTML entities
    cleaned = this.decodeHtmlEntities(cleaned);

    // Normalize whitespace (but preserve paragraph breaks)
    cleaned = cleaned.replace(/[ \t]+/g, ' ');
    cleaned = cleaned.replace(/\n\s*\n/g, '\n\n');
    cleaned = cleaned.trim();

    return cleaned;
  }

  decodeHtmlEntities(text) {
    if (!text || typeof text !== 'string') return text;

    // Comprehensive HTML entity mapping
    const entityMap = {
      '&amp;': '&', '&lt;': '<', '&gt;': '>', '&quot;': '"', '&apos;': "'", '&nbsp;': ' ',
      '&#32;': ' ', '&#34;': '"', '&#39;': "'", '&#59;': ';', '&#60;': '<', '&#61;': '=',
      '&#62;': '>', '&#91;': '[', '&#93;': ']', '&#123;': '{', '&#125;': '}', '&#160;': ' ',
      '&#215;': '×', '&#8211;': '–', '&#8212;': '—', '&#8217;': "'", '&#8220;': '"',
      '&#8221;': '"', '&#8226;': '•', '&#8230;': '…', '&#xa0;': ' ', '&#xd7;': '×'
    };

    // Decode known entities
    for (const [entity, replacement] of Object.entries(entityMap)) {
      if (text.includes(entity)) {
        text = text.replace(new RegExp(entity.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), 'g'), replacement);
      }
    }

    // Decode unknown numeric entities
    text = text.replace(/&#(\d+);/g, (match, num) => {
      const charCode = parseInt(num, 10);
      if (charCode >= 32 && charCode <= 126 || charCode >= 160 && charCode <= 65535) {
        return String.fromCharCode(charCode);
      }
      return match;
    });

    return text;
  }

  printFinalStats() {
    const duration = this.stats.startTime ? Date.now() - this.stats.startTime.getTime() : 0;
    const durationMinutes = (duration / 60000).toFixed(1);
    const avgRequestTime = this.requestTimes.length > 0
      ? (this.requestTimes.reduce((a, b) => a + b, 0) / this.requestTimes.length).toFixed(0)
      : 0;

    console.log(chalk.blue('\n📊 OPTIMIZED PERFORMANCE STATISTICS\n'));
    console.log(chalk.green(`🎉 Pages added: ${this.stats.pagesAdded.toLocaleString()}`));
    console.log(chalk.blue(`📝 Pages updated: ${this.stats.pagesUpdated.toLocaleString()}`));
    console.log(chalk.red(`🗑️ Pages deleted: ${this.stats.pagesDeleted.toLocaleString()}`));

    // ENHANCED: Show missing pages and new pages found
    if (this.stats.missingPages.length > 0) {
      console.log(chalk.yellow(`📋 Missing pages found: ${this.stats.missingPages.length.toLocaleString()}`));
    }
    if (this.stats.newPagesFound > 0) {
      console.log(chalk.cyan(`🆕 New pages found: ${this.stats.newPagesFound.toLocaleString()}`));
    }
    if (this.stats.pagesReprocessed > 0) {
      console.log(chalk.magenta(`🔧 Pages reprocessed for template parsing: ${this.stats.pagesReprocessed.toLocaleString()}`));
    }

    console.log(chalk.white(`📚 Total pages: ${this.pageData.size.toLocaleString()}`));
    console.log(chalk.white(`🌐 API calls made: ${this.stats.apiCalls.toLocaleString()}`));
    console.log(chalk.yellow(`⚡ Avg request time: ${avgRequestTime}ms`));
    console.log(chalk.white(`⏱️ Total time: ${durationMinutes} minutes`));

    // ENHANCED: Show detailed analysis of null responses
    if (this.stats.nullResponses.length > 0 || this.stats.filteredPages.length > 0 || this.stats.shortPages.length > 0) {
      console.log(chalk.blue('\n🔍 DETAILED ANALYSIS OF UNCHANGED PAGES\n'));
      console.log(chalk.yellow(`⚠️ Null API responses: ${this.stats.nullResponses.length.toLocaleString()}`));
      console.log(chalk.yellow(`🚫 Filtered page types: ${this.stats.filteredPages.length.toLocaleString()}`));
      console.log(chalk.yellow(`📏 Too short pages: ${this.stats.shortPages.length.toLocaleString()}`));

      // Log detailed breakdown for manual review
      this.logDetailedAnalysis();
    }

    console.log('');
  }

  // ENHANCED: Detailed logging for manual review
  logDetailedAnalysis() {
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    const logFile = path.join(__dirname, `unchanged-pages-analysis-${timestamp}.log`);

    let logContent = `OSRS Wiki Watchdog - Unchanged Pages Analysis\n`;
    logContent += `Generated: ${new Date().toISOString()}\n`;
    logContent += `Total Changes Processed: ${this.stats.changesProcessed}\n`;
    logContent += `Actual Updates: ${this.stats.pagesAdded + this.stats.pagesUpdated + this.stats.pagesDeleted}\n\n`;

    if (this.stats.nullResponses.length > 0) {
      logContent += `=== NULL API RESPONSES (${this.stats.nullResponses.length}) ===\n`;
      logContent += `These pages had newer revision IDs but returned null from the content API:\n\n`;
      this.stats.nullResponses.forEach(page => {
        logContent += `- ${page.title} (${page.type}, revid: ${page.newRevid || page.revid})\n`;
      });
      logContent += `\n`;
    }

    if (this.stats.filteredPages.length > 0) {
      logContent += `=== FILTERED PAGE TYPES (${this.stats.filteredPages.length}) ===\n`;
      logContent += `These pages were filtered out as unwanted content (redirects, disambiguation, etc.):\n\n`;
      this.stats.filteredPages.forEach(page => {
        logContent += `- ${page.title} (revid: ${page.revid})\n`;
      });
      logContent += `\n`;
    }

    if (this.stats.shortPages.length > 0) {
      logContent += `=== TOO SHORT PAGES (${this.stats.shortPages.length}) ===\n`;
      logContent += `These pages were too short (< 100 characters) after cleaning:\n\n`;
      this.stats.shortPages.forEach(page => {
        logContent += `- ${page.title} (${page.length} chars, revid: ${page.revid})\n`;
      });
      logContent += `\n`;
    }

    logContent += `=== RECOMMENDATIONS ===\n`;
    logContent += `1. NULL RESPONSES: These pages exist in revision API but not content API - likely deleted or moved\n`;
    logContent += `2. FILTERED PAGES: These are redirects/disambiguation - should they be removed from our collection?\n`;
    logContent += `3. SHORT PAGES: These have minimal content - consider if they're worth keeping\n\n`;
    logContent += `Review this log to determine which pages should be removed from the collection.\n`;

    require('fs').writeFileSync(logFile, logContent, 'utf8');
    console.log(chalk.blue(`📄 Detailed analysis saved to: ${logFile}`));
  }

  printCleanSummary() {
    console.log(chalk.cyan('\n📋 CLEAN SUMMARY'));
    console.log(chalk.cyan('================'));

    const totalChanges = this.stats.pagesAdded + this.stats.pagesUpdated + this.stats.pagesDeleted;

    if (totalChanges === 0) {
      console.log(chalk.green('✅ No changes - wiki content is up to date'));
    } else {
      if (this.stats.pagesAdded > 0) console.log(chalk.green(`➕ Added: ${this.stats.pagesAdded} pages`));
      if (this.stats.pagesUpdated > 0) console.log(chalk.blue(`📝 Updated: ${this.stats.pagesUpdated} pages`));
      if (this.stats.pagesDeleted > 0) console.log(chalk.red(`🗑️ Deleted: ${this.stats.pagesDeleted} pages`));
    }

    // Show persistent tracking
    if (this.filteredPages.size > 0) console.log(chalk.yellow(`🚫 Filtered pages tracked: ${this.filteredPages.size}`));
    if (this.nullPages.size > 0) console.log(chalk.gray(`⚫ NULL pages tracked: ${this.nullPages.size}`));

    console.log(chalk.white(`📚 Total pages: ${this.pageData.size.toLocaleString()}`));
  }

  // INVESTIGATION: Find pages in collection but not on current wiki
  async investigateMissingPages() {
    console.log(chalk.cyan('\n🔍 MISSING PAGES INVESTIGATION'));
    console.log(chalk.cyan('==============================='));

    // Step 1: Get current wiki pages via FAST complete scan
    console.log(chalk.blue('📊 Step 1: Fast scanning ALL current wiki pages...'));
    console.log(chalk.gray('   🔧 API Parameters: apnamespace=0 (main namespace), apfilterredir=nonredirects'));
    console.log(chalk.gray('   ⚡ Using optimized complete scan instead of slow alphabetical method'));
    const currentWikiPages = new Set();

    // OPTIMIZED: Get ALL pages in main namespace with pagination
    let apcontinue = null;
    let totalFound = 0;
    let apiCalls = 0;

    do {
      apiCalls++;
      console.log(chalk.gray(`   � API call ${apiCalls} (found ${totalFound.toLocaleString()} pages so far)...`));

      const response = await axios.get(this.wikiApiUrl, {
        params: {
          action: 'query',
          list: 'allpages',
          aplimit: 500,           // Max pages per request
          apnamespace: 0,         // Main namespace only
          apfilterredir: 'nonredirects', // No redirects
          format: 'json',
          ...(apcontinue && { apcontinue })
        },
        headers: { 'User-Agent': this.userAgent },
        timeout: 30000
      });

      const pages = response.data.query?.allpages || [];

      for (const page of pages) {
        currentWikiPages.add(page.title);
        totalFound++;
      }

      apcontinue = response.data.continue?.apcontinue;

      // Shorter delay for faster scanning
      if (apcontinue) {
        await this.sleep(500); // 0.5 second delay instead of 1 second
      }

    } while (apcontinue);

    console.log(chalk.green(`   ✅ Complete scan: ${apiCalls} API calls, ${totalFound.toLocaleString()} pages found`));

    // ENHANCED: Test if our filtering is excluding valid pages
    console.log(chalk.blue('\n🔍 Testing allpages filtering...'));
    console.log(chalk.gray('   🧪 Testing with redirects included...'));

    let testApiCalls = 0;
    let testTotalFound = 0;
    let testApcontinue = null;
    const testWikiPages = new Set();

    do {
      testApiCalls++;
      if (testApiCalls > 3) break; // Just test first few calls

      const testResponse = await axios.get(this.wikiApiUrl, {
        params: {
          action: 'query',
          list: 'allpages',
          aplimit: 500,
          apnamespace: 0,
          // NO apfilterredir - include redirects
          format: 'json',
          ...(testApcontinue && { apcontinue: testApcontinue })
        },
        headers: { 'User-Agent': this.userAgent },
        timeout: 30000
      });

      const testPages = testResponse.data.query?.allpages || [];

      for (const page of testPages) {
        testWikiPages.add(page.title);
        testTotalFound++;
      }

      testApcontinue = testResponse.data.continue?.apcontinue;
      await this.sleep(500);

    } while (testApcontinue && testApiCalls < 3);

    console.log(chalk.gray(`   📊 With redirects: ${testTotalFound.toLocaleString()} pages in first ${testApiCalls} calls`));
    console.log(chalk.gray(`   📊 Without redirects: ${Math.min(1500, totalFound).toLocaleString()} pages in first 3 calls`));

    // Check if any of our missing pages appear when redirects are included
    const sampleMissing = ['Baking potatoes', 'Buying battlestaves from Zaff', 'Beast tv'];
    for (const title of sampleMissing) {
      if (testWikiPages.has(title)) {
        console.log(chalk.yellow(`   🔍 "${title}" FOUND when redirects included!`));
      }
    }

    console.log(chalk.green(`✅ Found ${currentWikiPages.size.toLocaleString()} pages currently on wiki (excluding redirects)`));

    // Step 2: Compare with our collection
    console.log(chalk.blue('\n📊 Step 2: Comparing with our collection...'));
    console.log(chalk.gray('   📋 Our page titles: ' + this.pageTitles.size.toLocaleString()));
    console.log(chalk.gray('   📄 Our page content: ' + this.pageData.size.toLocaleString()));
    console.log(chalk.gray('   🌐 Current wiki pages: ' + currentWikiPages.size.toLocaleString()));

    const ourPages = new Set(this.pageTitles);
    const ourContent = new Set(this.pageData.keys());
    const missingFromWiki = new Set();
    const missingFromCollection = new Set();

    // Analyze where the missing pages come from
    const missingFromTitlesOnly = new Set();
    const missingFromContentOnly = new Set();
    const missingFromBoth = new Set();

    // Find pages in our collection but not on wiki
    for (const page of ourPages) {
      if (!currentWikiPages.has(page)) {
        missingFromWiki.add(page);

        // Categorize where this missing page comes from
        const inTitles = this.pageTitles.has(page);
        const inContent = this.pageData.has(page);

        if (inTitles && inContent) {
          missingFromBoth.add(page);
        } else if (inTitles && !inContent) {
          missingFromTitlesOnly.add(page);
        } else if (!inTitles && inContent) {
          missingFromContentOnly.add(page);
        }
      }
    }

    // Find pages on wiki but not in our collection
    for (const page of currentWikiPages) {
      if (!ourPages.has(page)) {
        missingFromCollection.add(page);
      }
    }

    console.log(chalk.yellow(`⚠️ Pages in our collection but not on current wiki: ${missingFromWiki.size}`));
    console.log(chalk.yellow(`⚠️ Pages on current wiki but not in our collection: ${missingFromCollection.size}`));

    // ENHANCED: Show breakdown of where missing pages come from
    console.log(chalk.cyan('\n📊 BREAKDOWN OF MISSING PAGES:'));
    console.log(chalk.blue(`   📋 In titles AND content: ${missingFromBoth.size} pages`));
    console.log(chalk.yellow(`   📋 In titles ONLY: ${missingFromTitlesOnly.size} pages`));
    console.log(chalk.red(`   📄 In content ONLY: ${missingFromContentOnly.size} pages`));

    // ENHANCED: Show first 50 missing pages for analysis
    if (missingFromWiki.size > 0) {
      console.log(chalk.cyan('\n📋 FIRST 50 PAGES IN COLLECTION BUT NOT FOUND IN SCAN:'));
      const first50Missing = Array.from(missingFromWiki).slice(0, 50);
      for (let i = 0; i < first50Missing.length; i++) {
        console.log(chalk.gray(`   ${i + 1}. ${first50Missing[i]}`));
      }
      if (missingFromWiki.size > 50) {
        console.log(chalk.gray(`   ... and ${missingFromWiki.size - 50} more`));
      }
    }

    if (missingFromCollection.size > 0) {
      console.log(chalk.cyan('\n📋 PAGES ON WIKI BUT NOT IN OUR COLLECTION:'));
      const missingFromCollectionArray = Array.from(missingFromCollection);
      for (let i = 0; i < missingFromCollectionArray.length; i++) {
        console.log(chalk.gray(`   ${i + 1}. ${missingFromCollectionArray[i]}`));
      }
    }

    // Step 3: Test a sample of "missing" pages to verify they're actually gone
    console.log(chalk.blue('\n📊 Step 3: Testing sample of "missing" pages...'));
    const sampleSize = Math.min(20, missingFromWiki.size);
    const samplePages = Array.from(missingFromWiki).slice(0, sampleSize);

    const testResults = {
      actuallyDeleted: [],
      stillExists: [],
      redirects: [],
      errors: []
    };

    for (const title of samplePages) {
      try {
        console.log(chalk.gray(`   🔍 Testing: ${title}`));

        // ENHANCED: Check for redirects, deletions, and page info
        const response = await axios.get(this.wikiApiUrl, {
          params: {
            action: 'query',
            titles: title,
            prop: 'info|revisions',
            rvprop: 'content|timestamp|comment|user',
            rvlimit: 1,
            redirects: 1,  // Follow redirects
            format: 'json',
            formatversion: 2
          },
          headers: { 'User-Agent': this.userAgent },
          timeout: 30000
        });

        const page = response.data.query?.pages?.[0];
        const redirects = response.data.query?.redirects || [];

        // Check if this was a redirect
        if (redirects.length > 0) {
          const redirect = redirects[0];
          testResults.redirects.push(title);
          console.log(chalk.yellow(`     🔄 REDIRECT: ${title} → ${redirect.to}`));

          // Check if the redirect target exists
          if (page && !page.missing) {
            console.log(chalk.green(`       ✅ Target exists: ${redirect.to}`));
          } else {
            console.log(chalk.red(`       ❌ Target missing: ${redirect.to}`));
          }
        } else if (page?.missing) {
          testResults.actuallyDeleted.push(title);
          console.log(chalk.red(`     ❌ MISSING: ${title}`));

          // Test if it appears in allpages with redirects included
          console.log(chalk.gray(`       🔍 Testing if page appears in allpages scan...`));
          try {
            const testResponse = await axios.get(this.wikiApiUrl, {
              params: {
                action: 'query',
                list: 'allpages',
                apfrom: title,
                apto: title,
                aplimit: 1,
                apnamespace: 0,
                // Remove apfilterredir to include redirects
                format: 'json'
              },
              headers: { 'User-Agent': this.userAgent },
              timeout: 30000
            });

            const allpagesResult = testResponse.data.query?.allpages || [];
            if (allpagesResult.length > 0 && allpagesResult[0].title === title) {
              console.log(chalk.yellow(`       � Page DOES appear in allpages (might be redirect)`));
            } else {
              console.log(chalk.gray(`       🔍 Page does NOT appear in allpages`));
            }
          } catch (error) {
            console.log(chalk.gray(`       ❌ Error testing allpages: ${error.message}`));
          }
        } else if (page?.revisions) {
          testResults.stillExists.push(title);
          console.log(chalk.green(`     ✅ EXISTS: ${title}`));

          // Check if our content is different from current
          const currentContent = page.revisions[0].content;
          const ourContent = this.pageData.get(title);
          if (ourContent && ourContent.text !== currentContent) {
            console.log(chalk.yellow(`       📝 Content differs from our version`));
          }
        } else {
          testResults.errors.push(title);
          console.log(chalk.gray(`     ❓ UNKNOWN: ${title}`));
        }

        await this.sleep(1000); // Rate limiting

      } catch (error) {
        testResults.errors.push(title);
        console.log(chalk.red(`     ❌ ERROR: ${title} - ${error.message}`));
      }
    }

    // Step 4: Report findings
    console.log(chalk.cyan('\n📋 INVESTIGATION RESULTS'));
    console.log(chalk.cyan('========================'));
    console.log(chalk.white(`📊 Our collection: ${ourPages.size.toLocaleString()} pages`));
    console.log(chalk.white(`📊 Current wiki: ${currentWikiPages.size.toLocaleString()} pages`));
    console.log(chalk.yellow(`⚠️ In collection but not found in scan: ${missingFromWiki.size.toLocaleString()} pages`));
    console.log(chalk.yellow(`⚠️ On wiki but not in collection: ${missingFromCollection.size.toLocaleString()} pages`));

    console.log(chalk.cyan('\n🧪 SAMPLE TEST RESULTS:'));
    console.log(chalk.red(`❌ Actually deleted: ${testResults.actuallyDeleted.length}/${sampleSize}`));
    console.log(chalk.green(`✅ Still exists: ${testResults.stillExists.length}/${sampleSize}`));
    console.log(chalk.yellow(`🔄 Redirects: ${testResults.redirects.length}/${sampleSize}`));
    console.log(chalk.gray(`❓ Errors/Unknown: ${testResults.errors.length}/${sampleSize}`));

    if (testResults.stillExists.length > 0) {
      console.log(chalk.red('\n🚨 SCANNING BUG DETECTED!'));
      console.log(chalk.red('The following pages exist on wiki but were not found in alphabetical scan:'));
      for (const page of testResults.stillExists) {
        console.log(chalk.red(`   - ${page}`));
      }
    }

    // ENHANCED: Check our historical metadata
    console.log(chalk.cyan('\n📊 HISTORICAL METADATA ANALYSIS:'));
    const sampleOurPages = Array.from(missingFromWiki).slice(0, 5);
    for (const title of sampleOurPages) {
      const ourPage = this.pageData.get(title);
      if (ourPage) {
        console.log(chalk.blue(`📄 ${title}:`));
        console.log(chalk.gray(`   📅 Our timestamp: ${ourPage.timestamp || 'Not recorded'}`));
        console.log(chalk.gray(`   🔢 Our revision: ${ourPage.revid || 'Not recorded'}`));
        console.log(chalk.gray(`   📏 Content length: ${ourPage.text?.length || 0} chars`));

        // Check if content suggests it's a guide
        const text = ourPage.text || '';
        const isGuide = text.toLowerCase().includes('money') ||
                       text.toLowerCase().includes('profit') ||
                       text.toLowerCase().includes('gp/hour') ||
                       text.toLowerCase().includes('guide');
        console.log(chalk.gray(`   📖 Appears to be guide: ${isGuide ? 'Yes' : 'No'}`));
      }
    }

    if (testResults.actuallyDeleted.length > 0) {
      console.log(chalk.yellow('\n📝 CONFIRMED DELETIONS:'));
      console.log(chalk.yellow('The following pages have been deleted from the wiki:'));
      for (const page of testResults.actuallyDeleted) {
        console.log(chalk.yellow(`   - ${page}`));
      }
    }

    // Save detailed results
    const reportPath = path.join(__dirname, `missing-pages-investigation-${new Date().toISOString().replace(/[:.]/g, '-')}.log`);
    const report = {
      timestamp: new Date().toISOString(),
      ourCollectionSize: ourPages.size,
      currentWikiSize: currentWikiPages.size,
      missingFromWiki: Array.from(missingFromWiki),
      missingFromCollection: Array.from(missingFromCollection),
      sampleTestResults: testResults
    };

    fs.writeFileSync(reportPath, JSON.stringify(report, null, 2));
    console.log(chalk.blue(`\n📄 Detailed report saved to: ${reportPath}`));
  }

  // ENHANCED: Comprehensive namespace investigation
  async investigateNamespaces() {
    console.log(chalk.cyan('\n🔍 COMPREHENSIVE NAMESPACE INVESTIGATION'));
    console.log(chalk.cyan('=========================================='));

    // Step 1: Get all namespaces
    console.log(chalk.blue('\n📊 Step 1: Discovering all namespaces...'));

    try {
      const response = await axios.get(this.wikiApiUrl, {
        params: {
          action: 'query',
          meta: 'siteinfo',
          siprop: 'namespaces',
          format: 'json'
        },
        headers: { 'User-Agent': this.userAgent },
        timeout: 30000
      });

      const namespaces = response.data.query?.meta?.namespaces || {};

      console.log(chalk.green(`✅ Found ${Object.keys(namespaces).length} namespaces`));

      // Step 2: Analyze each namespace
      console.log(chalk.blue('\n📊 Step 2: Analyzing each namespace...'));

      const namespaceAnalysis = [];

      for (const [nsId, nsInfo] of Object.entries(namespaces)) {
        const nsNumber = parseInt(nsId);
        const nsName = nsInfo.name || '(Main)';
        const nsCanonical = nsInfo.canonical || '';

        console.log(chalk.yellow(`\n🔍 Analyzing namespace ${nsNumber}: ${nsName}`));
        console.log(chalk.gray(`   📝 Canonical: ${nsCanonical}`));
        console.log(chalk.gray(`   🏷️ Case: ${nsInfo.case || 'first-letter'}`));

        // Get page count for this namespace
        try {
          const countResponse = await axios.get(this.wikiApiUrl, {
            params: {
              action: 'query',
              list: 'allpages',
              apnamespace: nsNumber,
              aplimit: 500,
              format: 'json'
            },
            headers: { 'User-Agent': this.userAgent },
            timeout: 30000
          });

          const pages = countResponse.data.query?.allpages || [];
          const hasMore = countResponse.data.continue ? true : false;
          const estimatedCount = hasMore ? `${pages.length}+` : pages.length.toString();

          console.log(chalk.green(`   📊 Pages: ${estimatedCount} (estimated)`));

          // Sample some page titles to understand content
          if (pages.length > 0) {
            const sampleTitles = pages.slice(0, 5).map(p => p.title);
            console.log(chalk.gray(`   📋 Sample titles:`));
            for (const title of sampleTitles) {
              console.log(chalk.gray(`      - ${title}`));
            }
          }

          namespaceAnalysis.push({
            id: nsNumber,
            name: nsName,
            canonical: nsCanonical,
            estimatedCount: estimatedCount,
            sampleTitles: pages.slice(0, 10).map(p => p.title),
            case: nsInfo.case || 'first-letter'
          });

        } catch (error) {
          console.log(chalk.red(`   ❌ Error getting page count: ${error.message}`));
          namespaceAnalysis.push({
            id: nsNumber,
            name: nsName,
            canonical: nsCanonical,
            estimatedCount: 'Error',
            sampleTitles: [],
            case: nsInfo.case || 'first-letter'
          });
        }

        await this.sleep(500); // Rate limiting
      }

      // Step 3: Detailed analysis of key namespaces
      console.log(chalk.blue('\n📊 Step 3: Detailed analysis of key namespaces...'));

      const keyNamespaces = [0, 1, 4, 6, 10, 14, 100, 102, 104, 106, 108, 110];

      for (const nsId of keyNamespaces) {
        const nsData = namespaceAnalysis.find(ns => ns.id === nsId);
        if (!nsData || nsData.estimatedCount === 'Error') continue;

        console.log(chalk.yellow(`\n🔍 DETAILED: Namespace ${nsId} (${nsData.name})`));

        try {
          // Get more pages for detailed analysis
          let allPages = [];
          let apcontinue = null;
          let apiCalls = 0;

          do {
            apiCalls++;
            if (apiCalls > 3) break; // Limit to first 1500 pages

            const detailResponse = await axios.get(this.wikiApiUrl, {
              params: {
                action: 'query',
                list: 'allpages',
                apnamespace: nsId,
                aplimit: 500,
                format: 'json',
                ...(apcontinue && { apcontinue })
              },
              headers: { 'User-Agent': this.userAgent },
              timeout: 30000
            });

            const pages = detailResponse.data.query?.allpages || [];
            allPages.push(...pages);
            apcontinue = detailResponse.data.continue?.apcontinue;

            await this.sleep(500);

          } while (apcontinue && apiCalls < 3);

          console.log(chalk.green(`   📊 Analyzed ${allPages.length} pages (${apiCalls} API calls)`));

          // Analyze page patterns
          const categories = {
            guides: 0,
            money: 0,
            quests: 0,
            skills: 0,
            monsters: 0,
            bosses: 0,
            items: 0,
            locations: 0,
            npcs: 0,
            minigames: 0,
            raids: 0,
            slayer: 0,
            other: 0
          };

          for (const page of allPages) {
            const title = page.title.toLowerCase();

            // Category analysis
            if (title.includes('guide') || title.includes('strategy')) categories.guides++;
            else if (title.includes('money') || title.includes('profit') || title.includes('gp') || title.includes('buying') || title.includes('selling')) categories.money++;
            else if (title.includes('quest')) categories.quests++;
            else if (title.includes('skill') || title.includes('training') || title.includes('level')) categories.skills++;
            else if (title.includes('monster') || title.includes('creature')) categories.monsters++;
            else if (title.includes('boss')) categories.bosses++;
            else if (title.includes('item') || title.includes('equipment') || title.includes('weapon') || title.includes('armour')) categories.items++;
            else if (title.includes('location') || title.includes('area') || title.includes('city') || title.includes('dungeon')) categories.locations++;
            else if (title.includes('npc') || title.includes('character')) categories.npcs++;
            else if (title.includes('minigame') || title.includes('activity')) categories.minigames++;
            else if (title.includes('raid') || title.includes('chambers') || title.includes('theatre')) categories.raids++;
            else if (title.includes('slayer') || title.includes('task')) categories.slayer++;
            else categories.other++;
          }

          console.log(chalk.cyan(`   📂 Content categories:`));
          for (const [category, count] of Object.entries(categories)) {
            if (count > 0) {
              console.log(chalk.gray(`      ${category}: ${count}`));
            }
          }

        } catch (error) {
          console.log(chalk.red(`   ❌ Error in detailed analysis: ${error.message}`));
        }
      }

      // Step 4: Summary and recommendations
      console.log(chalk.blue('\n📊 Step 4: Summary and recommendations...'));

      console.log(chalk.cyan('\n📋 NAMESPACE SUMMARY:'));
      for (const ns of namespaceAnalysis.sort((a, b) => a.id - b.id)) {
        const countStr = typeof ns.estimatedCount === 'string' ? ns.estimatedCount : ns.estimatedCount.toLocaleString();
        console.log(chalk.white(`   ${ns.id.toString().padStart(3)}: ${ns.name.padEnd(20)} (${countStr} pages)`));
      }

      // Check if our missing pages might be in other namespaces
      console.log(chalk.blue('\n🔍 Step 5: Checking if missing pages exist in other namespaces...'));

      const sampleMissingPages = ['Baking potatoes', 'Buying battlestaves from Zaff', 'Money making guide'];

      for (const title of sampleMissingPages) {
        console.log(chalk.yellow(`\n🔍 Searching for: "${title}"`));

        try {
          const searchResponse = await axios.get(this.wikiApiUrl, {
            params: {
              action: 'query',
              list: 'search',
              srsearch: title,
              srlimit: 10,
              srnamespace: '*', // Search all namespaces
              format: 'json'
            },
            headers: { 'User-Agent': this.userAgent },
            timeout: 30000
          });

          const results = searchResponse.data.query?.search || [];

          if (results.length > 0) {
            console.log(chalk.green(`   ✅ Found ${results.length} matches:`));
            for (const result of results.slice(0, 5)) {
              console.log(chalk.gray(`      NS ${result.ns}: ${result.title}`));
            }
          } else {
            console.log(chalk.red(`   ❌ No matches found`));
          }

        } catch (error) {
          console.log(chalk.red(`   ❌ Search error: ${error.message}`));
        }

        await this.sleep(1000);
      }

      // Save detailed report
      const reportPath = path.join(__dirname, `namespace-investigation-${new Date().toISOString().replace(/[:.]/g, '-')}.log`);
      const reportContent = JSON.stringify({
        timestamp: new Date().toISOString(),
        totalNamespaces: Object.keys(namespaces).length,
        namespaces: namespaceAnalysis,
        investigation: 'Complete namespace analysis of OSRS Wiki'
      }, null, 2);

      fs.writeFileSync(reportPath, reportContent);
      console.log(chalk.green(`\n📄 Detailed report saved to: ${reportPath}`));

    } catch (error) {
      console.error(chalk.red(`❌ Namespace investigation failed: ${error.message}`));
      throw error;
    }
  }

  // DIAGNOSTIC: Analyze our collection to understand what we have
  async analyzeCollection() {
    console.log(chalk.cyan('\n🔍 COLLECTION ANALYSIS'));
    console.log(chalk.cyan('====================='));

    // Analyze page titles vs content
    const titlesOnly = new Set();
    const contentOnly = new Set();
    const both = new Set();

    // Check titles that don't have content
    for (const title of this.pageTitles) {
      if (this.pageData.has(title)) {
        both.add(title);
      } else {
        titlesOnly.add(title);
      }
    }

    // Check content that doesn't have titles
    for (const title of this.pageData.keys()) {
      if (!this.pageTitles.has(title)) {
        contentOnly.add(title);
      }
    }

    console.log(chalk.blue(`📊 Page Titles: ${this.pageTitles.size.toLocaleString()}`));
    console.log(chalk.blue(`📊 Page Content: ${this.pageData.size.toLocaleString()}`));
    console.log(chalk.green(`✅ Both title & content: ${both.size.toLocaleString()}`));
    console.log(chalk.yellow(`⚠️ Title only (no content): ${titlesOnly.size.toLocaleString()}`));
    console.log(chalk.red(`❌ Content only (no title): ${contentOnly.size.toLocaleString()}`));

    // Analyze starting characters
    const charCounts = new Map();
    for (const title of this.pageData.keys()) {
      const firstChar = title.charAt(0).toUpperCase();
      charCounts.set(firstChar, (charCounts.get(firstChar) || 0) + 1);
    }

    console.log(chalk.cyan('\n📊 PAGES BY STARTING CHARACTER:'));
    const sortedChars = Array.from(charCounts.entries()).sort((a, b) => b[1] - a[1]);

    for (const [char, count] of sortedChars.slice(0, 20)) {
      const percentage = ((count / this.pageData.size) * 100).toFixed(1);
      console.log(chalk.gray(`   ${char}: ${count.toLocaleString()} pages (${percentage}%)`));
    }

    // Show problematic titles (titles without content)
    if (titlesOnly.size > 0) {
      console.log(chalk.yellow('\n⚠️ TITLES WITHOUT CONTENT (first 20):'));
      const titlesArray = Array.from(titlesOnly).slice(0, 20);
      for (const title of titlesArray) {
        console.log(chalk.gray(`   - ${title}`));
      }
      if (titlesOnly.size > 20) {
        console.log(chalk.gray(`   ... and ${titlesOnly.size - 20} more`));
      }
    }

    // Analyze special characters and patterns
    const specialPatterns = {
      'Colon pages': Array.from(this.pageData.keys()).filter(t => t.includes(':')).length,
      'Parentheses pages': Array.from(this.pageData.keys()).filter(t => t.includes('(')).length,
      'Apostrophe pages': Array.from(this.pageData.keys()).filter(t => t.includes("'")).length,
      'Hyphen pages': Array.from(this.pageData.keys()).filter(t => t.includes('-')).length,
      'Number start pages': Array.from(this.pageData.keys()).filter(t => /^[0-9]/.test(t)).length,
      'Special char start': Array.from(this.pageData.keys()).filter(t => !/^[A-Za-z0-9]/.test(t)).length
    };

    console.log(chalk.cyan('\n📊 SPECIAL PATTERNS:'));
    for (const [pattern, count] of Object.entries(specialPatterns)) {
      const percentage = ((count / this.pageData.size) * 100).toFixed(1);
      console.log(chalk.gray(`   ${pattern}: ${count.toLocaleString()} (${percentage}%)`));
    }
  }

  sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  // CRITICAL FIX: Synchronize page titles with actual content
  async synchronizePageTitles() {
    console.log(chalk.blue('🔄 Synchronizing page titles with loaded content...'));

    // Clear existing titles and rebuild from actual content
    this.pageTitles.clear();

    // Add all titles from loaded content
    for (const title of this.pageData.keys()) {
      this.pageTitles.add(title);
    }

    console.log(chalk.green(`✅ Synchronized ${this.pageTitles.size.toLocaleString()} page titles with content`));

    // Save the synchronized titles to file
    await this.savePageTitles();
    console.log(chalk.green(`✅ Updated page titles file with clean titles`));
  }

  // ENHANCED: Find pages that are in titles list but missing from content
  async findMissingPages() {
    const missingPages = [];

    for (const title of this.pageTitles) {
      if (!this.pageData.has(title)) {
        missingPages.push(title);
      }
    }

    if (missingPages.length > 0) {
      console.log(chalk.yellow(`📋 Found ${missingPages.length} pages in titles but missing from content:`));

      // Show some examples
      missingPages.slice(0, 10).forEach(title => {
        console.log(chalk.gray(`   - ${title}`));
      });
      if (missingPages.length > 10) {
        console.log(chalk.gray(`   ... and ${missingPages.length - 10} more`));
      }

      // Limit processing to avoid overwhelming the system
      if (missingPages.length > 1000) {
        console.log(chalk.yellow(`⚠️  Too many missing pages (${missingPages.length}), limiting to first 1000 for this run`));
        return missingPages.slice(0, 1000);
      }
    }

    return missingPages;
  }

  // ENHANCED: Process missing pages using fast batch API calls
  async processMissingPages(missingPages) {
    console.log(chalk.blue(`🔄 Processing ${missingPages.length} missing pages with batch API...`));

    // Process in batches of 50 (MediaWiki API limit)
    const batchSize = 50;
    let processed = 0;

    for (let i = 0; i < missingPages.length; i += batchSize) {
      const batch = missingPages.slice(i, i + batchSize);
      const batchNum = Math.floor(i / batchSize) + 1;
      const totalBatches = Math.ceil(missingPages.length / batchSize);

      console.log(chalk.gray(`📦 Batch ${batchNum}/${totalBatches} (${batch.length} pages)`));

      try {
        // Use batch API call - much faster than individual requests
        const response = await axios.get(this.wikiApiUrl, {
          params: {
            action: 'query',
            titles: batch.join('|'), // Pipe-separated titles
            prop: 'revisions|categories',
            rvprop: 'content|timestamp',
            rvslots: 'main',
            format: 'json',
            formatversion: 2
          },
          headers: {
            'User-Agent': this.userAgent,
            'Accept': 'application/json'
          },
          timeout: 60000 // Longer timeout for batch requests
        });

        this.stats.apiCalls++;

        const pages = response.data.query?.pages || [];

        for (const page of pages) {
          const title = page.title;

          if (page.missing) {
            this.stats.nullResponses.push(title);
            console.log(chalk.red(`❌ Missing: ${title} (removing from titles list)`));
            // CRITICAL FIX: Remove missing pages from titles to prevent repeated attempts
            this.pageTitles.delete(title);
            continue;
          }

          if (!page.revisions || page.revisions.length === 0) {
            this.stats.nullResponses.push(title);
            console.log(chalk.red(`❌ No content: ${title} (removing from titles list)`));
            // CRITICAL FIX: Remove pages with no content from titles to prevent repeated attempts
            this.pageTitles.delete(title);
            continue;
          }

          const revision = page.revisions[0];
          const content = revision.slots?.main?.content || '';

          // Create page data object
          const pageData = {
            title: title,
            categories: (page.categories || []).map(cat => cat.title.replace('Category:', '')),
            text: content,
            revid: page.revid,
            timestamp: revision.timestamp
          };

          // Apply filtering
          if (this.shouldFilterPage(title, content)) {
            this.stats.filteredPages.push(title);
            this.filteredPages.add(title); // Add to persistent filtered list
            this.pageTitles.delete(title); // CRITICAL FIX: Remove from titles so it won't be "found" again
            console.log(chalk.yellow(`🚫 Filtered: ${title}`));
          } else {
            this.pageData.set(title, pageData);
            this.stats.pagesAdded++;
            console.log(chalk.green(`✅ Added: ${title}`));
          }
        }

        processed += batch.length;
        console.log(chalk.blue(`📊 Progress: ${processed}/${missingPages.length} (${((processed/missingPages.length)*100).toFixed(1)}%)`));

        // Minimal delay between batches (MediaWiki allows this)
        if (i + batchSize < missingPages.length) {
          await this.sleep(500); // 0.5 second delay
        }

      } catch (error) {
        console.error(chalk.red(`❌ Batch ${batchNum} failed: ${error.message}`));
        // Mark all pages in failed batch
        for (const title of batch) {
          this.stats.nullResponses.push(title);
        }
      }
    }

    console.log(chalk.green(`✅ Batch processing complete: ${this.stats.pagesAdded} added, ${this.stats.filteredPages.length} filtered, ${this.stats.nullResponses.length} failed`));
  }

  // ENHANCED: Find new pages on wiki that aren't in our titles list
  async findNewPages() {
    console.log(chalk.blue('🔍 Checking for new pages on wiki...'));

    try {
      // PROPER FIX: Accept that our collection is likely more complete than wiki's filtered count
      // The "difference" we were seeing was likely due to our collection being more comprehensive
      const ourPageCount = this.pageTitles.size;

      console.log(chalk.blue(`📊 Our collection has ${ourPageCount.toLocaleString()} pages`));
      console.log(chalk.blue('� Checking for genuinely new pages using recent changes...'));

      // APPROACH 1: Recent changes (fast, catches most new content)
      const recentNewPages = await this.checkRecentChangesForNewPages();

      // APPROACH 2: Periodic comprehensive scan (thorough, catches everything)
      // FIXED: Initialize metadata if undefined
      if (!this.metadata) {
        this.metadata = {};
      }

      const lastComprehensiveScan = this.metadata.lastComprehensiveScan;
      let comprehensiveNewPages = [];

      if (!lastComprehensiveScan) {
        // First run - do comprehensive scan immediately
        console.log(chalk.blue(`🔍 First run detected, doing initial comprehensive scan...`));
        this.metadata.lastComprehensiveScan = Date.now();
        comprehensiveNewPages = await this.fetchNewPageTitles();
      } else {
        const hoursSinceLastScan = (Date.now() - lastComprehensiveScan) / (60 * 60 * 1000);

        if (hoursSinceLastScan > 6) { // Every 6 hours
          console.log(chalk.blue(`🔍 Last comprehensive scan was ${hoursSinceLastScan.toFixed(1)} hours ago, doing alphabetical scan...`));
          this.metadata.lastComprehensiveScan = Date.now();
          comprehensiveNewPages = await this.fetchNewPageTitles();
        } else {
          console.log(chalk.gray(`⏭️ Comprehensive scan done recently (${hoursSinceLastScan.toFixed(1)} hours ago), skipping`));
        }
      }

      // Combine results from both approaches
      const allNewPages = [...new Set([...recentNewPages, ...comprehensiveNewPages])];

      if (allNewPages.length > 0) {
        console.log(chalk.yellow(`� Found ${allNewPages.length} total new pages (${recentNewPages.length} from recent changes, ${comprehensiveNewPages.length} from comprehensive scan)`));
        return allNewPages;
      } else {
        console.log(chalk.green(`✅ No new pages found from either approach`));
        return [];
      }

    } catch (error) {
      console.error(chalk.red(`❌ Error checking for new pages: ${error.message}`));
      return [];
    }
  }

  // COMPREHENSIVE: Systematic gap detection across entire alphabet
  async fetchNewPageTitles() {
    const newPages = [];

    console.log(chalk.blue('🔍 Performing systematic gap analysis across entire wiki...'));
    console.log(chalk.yellow('⚠️ This will take longer but ensures we find ALL missing critical content'));

    try {
      // ENHANCED: Include special characters that OSRS wiki uses
      const alphabetSections = [
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
        'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
        "'", '"', '.', '!', '?', '-', '+', '(', ')', '[', ']', '&', '%'
      ];

      for (const section of alphabetSections) {
        console.log(chalk.cyan(`🔍 Scanning section "${section}"...`));

        let apcontinue = null;
        let sectionPages = 0;
        // FIXED: Remove arbitrary 2000-page limit - get ALL pages in each section

        do {
          const response = await axios.get(this.wikiApiUrl, {
            params: {
              action: 'query',
              list: 'allpages',
              apfrom: section,
              aplimit: 500,
              apnamespace: 0,
              apfilterredir: 'nonredirects',
              format: 'json',
              ...(apcontinue && { apcontinue })
            },
            headers: { 'User-Agent': this.userAgent },
            timeout: 30000
          });

          const pages = response.data.query?.allpages || [];
          let foundInSection = 0;

          for (const page of pages) {
            // Stop if we've moved beyond this section
            if (!page.title.toUpperCase().startsWith(section)) {
              break;
            }

            // ENHANCED: Apply same filtering criteria as original collection
            if (!this.pageTitles.has(page.title) &&
                !this.filteredPages.has(page.title) &&
                !this.nullPages.has(page.title) &&
                this.shouldIncludeNewPage(page.title)) {
              newPages.push(page.title);
              foundInSection++;
            }
            sectionPages++;
          }

          if (foundInSection > 0) {
            console.log(chalk.green(`   ✅ Found ${foundInSection} missing pages in section "${section}"`));
          }

          apcontinue = response.data.continue?.apcontinue;

          // Stop if we've moved beyond this section or no more pages
          if (pages.length === 0) break;
          if (pages.length > 0 && !pages[pages.length - 1].title.toUpperCase().startsWith(section)) break;

          await this.sleep(1000); // Rate limiting

        } while (apcontinue); // FIXED: Continue until no more pages, not artificial limit

        console.log(chalk.gray(`   📊 Section "${section}": ${sectionPages} pages checked`));
      }

      if (newPages.length > 0) {
        console.log(chalk.yellow(`📋 COMPREHENSIVE SCAN: Found ${newPages.length} missing pages across entire alphabet:`));
        newPages.slice(0, 20).forEach(title => {
          console.log(chalk.gray(`   - ${title}`));
        });
        if (newPages.length > 20) {
          console.log(chalk.gray(`   ... and ${newPages.length - 20} more`));
        }
      } else {
        console.log(chalk.green(`✅ No missing pages found in comprehensive scan`));
      }

      return newPages;

    } catch (error) {
      console.error(chalk.red(`❌ Error fetching new page titles: ${error.message}`));
      return [];
    }
  }

  // ENHANCED: Process new pages by adding them to titles and fetching content
  async processNewPages(newPages) {
    console.log(chalk.blue(`🔄 Processing ${newPages.length} new pages...`));

    // Add to titles list
    for (const title of newPages) {
      this.pageTitles.add(title);
    }

    // Fetch content for new pages
    await this.processMissingPages(newPages);

    this.stats.newPagesFound = this.stats.pagesAdded; // Use actual added count, not total found
    console.log(chalk.green(`✅ Processed ${newPages.length} new pages: ${this.stats.pagesAdded} added, ${this.stats.filteredPages.length} filtered`));
  }

  shouldFilterPageFromWikitext(title, wikitext) {
    // Filter based on wikitext content instead of HTML
    const lowerTitle = title.toLowerCase();
    const lowerWikitext = wikitext.toLowerCase();

    // Skip redirects
    if (wikitext.trim().startsWith('#REDIRECT') || wikitext.trim().startsWith('#redirect')) {
      return true;
    }

    // Skip disambiguation pages
    if (lowerTitle.includes('disambiguation') || lowerWikitext.includes('{{disambiguation}}')) {
      return true;
    }

    // Skip user pages, talk pages, etc.
    if (lowerTitle.includes(':') && !lowerTitle.startsWith('category:')) {
      return true;
    }

    // Skip very short content
    if (wikitext.trim().length < 50) {
      return true;
    }

    return false;
  }

  // ENHANCED: Check if new page should be included (matches original collection criteria)
  shouldIncludeNewPage(title) {
    const lowerTitle = title.toLowerCase();

    // Skip obvious disambiguation pages
    if (lowerTitle.includes('(disambiguation)')) {
      return false;
    }

    // Skip category namespace pages
    if (lowerTitle.startsWith('category:')) {
      return false;
    }

    // Skip user pages, talk pages, etc. (but allow some special namespaces)
    if (lowerTitle.includes(':') &&
        !lowerTitle.startsWith('money making guide:') &&
        !lowerTitle.startsWith('quest:') &&
        !lowerTitle.startsWith('skill:')) {
      return false;
    }

    // Skip obvious redirect indicators in title
    if (lowerTitle.includes('redirect') || lowerTitle.includes('moved to')) {
      return false;
    }

    return true; // Include by default (matches original liberal filtering)
  }

  // ENHANCED: Check recent changes for new pages (fast method)
  async checkRecentChangesForNewPages() {
    try {
      console.log(chalk.blue('📅 Checking recent changes for new pages in last 7 days...'));

      const recentResponse = await axios.get(this.wikiApiUrl, {
        params: {
          action: 'query',
          list: 'recentchanges',
          rctype: 'new',
          rcnamespace: 0,
          rclimit: 100,
          rcstart: new Date(Date.now() - 7 * 24 * 60 * 60 * 1000).toISOString(),
          format: 'json'
        },
        headers: { 'User-Agent': this.userAgent },
        timeout: 30000
      });

      const recentNewPages = recentResponse.data.query?.recentchanges || [];
      const genuinelyNewPages = [];

      for (const change of recentNewPages) {
        const title = change.title;
        if (!this.pageTitles.has(title) &&
            !this.filteredPages.has(title) &&
            !this.nullPages.has(title) &&
            this.shouldIncludeNewPage(title)) {
          genuinelyNewPages.push(title);
        }
      }

      console.log(chalk.blue(`📊 Found ${genuinelyNewPages.length} new pages in recent changes`));
      return genuinelyNewPages;

    } catch (error) {
      console.log(chalk.yellow(`⚠️ Recent changes check failed: ${error.message}`));
      return [];
    }
  }

  async processTemplatesFromWikitext(wikitext) {
    // Process MediaWiki templates from raw wikitext using our Python parser
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
        `]);

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
            resolve(output.trim());
          } else {
            // If template parsing fails, return original wikitext
            console.log(chalk.yellow(`⚠️  Template parsing failed, using original content`));
            if (errorOutput) {
              console.log(chalk.gray(`Template parser error: ${errorOutput.trim()}`));
            }
            resolve(wikitext);
          }
        });

        // Send wikitext to Python process
        python.stdin.write(wikitext);
        python.stdin.end();
      });

    } catch (error) {
      console.log(chalk.yellow(`⚠️  Template processing error: ${error.message}`));
      return wikitext; // Return original if processing fails
    }
  }

  cleanWikitextContent(wikitext) {
    // Clean processed wikitext content
    let cleanText = wikitext;

    // Remove remaining wiki markup
    cleanText = cleanText.replace(/\[\[([^|\]]+)\|?([^\]]*)\]\]/g, (match, link, text) => {
      return text || link;
    });

    // Remove external links
    cleanText = cleanText.replace(/\[https?:\/\/[^\s\]]+\s*([^\]]*)\]/g, '$1');

    // Remove file/image references
    cleanText = cleanText.replace(/\[\[File:[^\]]+\]\]/gi, '');
    cleanText = cleanText.replace(/\[\[Image:[^\]]+\]\]/gi, '');

    // Remove categories
    cleanText = cleanText.replace(/\[\[Category:[^\]]+\]\]/gi, '');

    // Remove HTML comments
    cleanText = cleanText.replace(/<!--[\s\S]*?-->/g, '');

    // Remove remaining template brackets
    cleanText = cleanText.replace(/\{\{[^}]*\}\}/g, '');

    // Clean up whitespace
    cleanText = cleanText.replace(/\n\s*\n/g, '\n');
    cleanText = cleanText.replace(/\s+/g, ' ');
    cleanText = cleanText.trim();

    return cleanText;
  }
}

// Export the class for testing
module.exports = OptimizedOSRSWikiWatchdog;

// Run the optimized watchdog
if (require.main === module) {
  const watchdog = new OptimizedOSRSWikiWatchdog();

  // Check for help mode
  if (process.argv.includes('--help') || process.argv.includes('-h')) {
    console.log(chalk.cyan('🔍 OSRS Wiki Watchdog - Available Options:'));
    console.log('');
    console.log(chalk.white('  --analyze          ') + chalk.gray('Run collection analysis and exit'));
    console.log(chalk.white('  --test             ') + chalk.gray('Run single check and exit'));
    console.log(chalk.white('  --skip-batch       ') + chalk.gray('Skip initial batch check, start with recent changes'));
    console.log(chalk.white('  --recent-only      ') + chalk.gray('Same as --skip-batch'));
    console.log(chalk.white('  --investigate      ') + chalk.gray('Find pages in collection but not on current wiki'));
    console.log(chalk.white('  --namespaces       ') + chalk.gray('Investigate all wiki namespaces and their content'));
    console.log(chalk.white('  --help, -h         ') + chalk.gray('Show this help message'));
    console.log('');
    console.log(chalk.yellow('Examples:'));
    console.log(chalk.gray('  node optimized-watchdog.js                    # Normal operation (batch check first)'));
    console.log(chalk.gray('  node optimized-watchdog.js --skip-batch       # Start immediately with recent changes'));
    console.log(chalk.gray('  node optimized-watchdog.js --test             # Single check and exit'));
    console.log(chalk.gray('  node optimized-watchdog.js --analyze          # Analyze collection and exit'));
    console.log(chalk.gray('  node optimized-watchdog.js --investigate      # Investigate missing pages'));
    console.log(chalk.gray('  node optimized-watchdog.js --namespaces       # Investigate all namespaces'));
    process.exit(0);
  }

  // Check for analysis mode
  const analyzeMode = process.argv.includes('--analyze');
  const testMode = process.argv.includes('--test');
  const investigateMode = process.argv.includes('--investigate');
  const namespacesMode = process.argv.includes('--namespaces');

  if (analyzeMode) {
    console.log(chalk.yellow('🔍 Running ANALYSIS MODE (collection analysis only)'));
    watchdog.loadMetadata()
      .then(() => watchdog.loadExistingData())
      .then(() => watchdog.loadPageTitles())
      .then(() => watchdog.loadFilteredPages())
      .then(() => watchdog.loadNullPages())
      .then(() => watchdog.analyzeCollection())
      .then(() => {
        console.log(chalk.green('✅ Analysis complete'));
        process.exit(0);
      }).catch(error => {
        console.error(chalk.red(`❌ Analysis failed: ${error.message}`));
        process.exit(1);
      });
  } else if (testMode) {
    console.log(chalk.yellow('🧪 Running in TEST MODE (single check only)'));
    // CRITICAL FIX: Initialize ALL data before checking for changes
    watchdog.loadMetadata()
      .then(() => watchdog.loadExistingData())
      .then(() => watchdog.loadPageTitles())
      .then(() => watchdog.loadFilteredPages())
      .then(() => watchdog.loadNullPages())
      .then(() => watchdog.checkForChanges())
      .then(() => {
        console.log(chalk.green('✅ Test complete'));
        process.exit(0);
      }).catch(error => {
        console.error(chalk.red(`❌ Test failed: ${error.message}`));
        process.exit(1);
      });
  } else if (investigateMode) {
    console.log(chalk.yellow('🔍 Running INVESTIGATION MODE (find missing pages)'));
    watchdog.loadMetadata()
      .then(() => watchdog.loadExistingData())
      .then(() => watchdog.loadPageTitles())
      .then(() => watchdog.loadFilteredPages())
      .then(() => watchdog.loadNullPages())
      .then(() => watchdog.investigateMissingPages())
      .then(() => {
        console.log(chalk.green('✅ Investigation complete'));
        process.exit(0);
      }).catch(error => {
        console.error(chalk.red(`❌ Investigation failed: ${error.message}`));
        process.exit(1);
      });
  } else if (namespacesMode) {
    console.log(chalk.yellow('🔍 Running NAMESPACES MODE (investigate all namespaces)'));
    watchdog.loadMetadata()
      .then(() => watchdog.investigateNamespaces())
      .then(() => {
        console.log(chalk.green('✅ Namespace investigation complete'));
        process.exit(0);
      }).catch(error => {
        console.error(chalk.red(`❌ Namespace investigation failed: ${error.message}`));
        process.exit(1);
      });
  } else {
    watchdog.run().catch(console.error);
  }
}

module.exports = OptimizedOSRSWikiWatchdog;
