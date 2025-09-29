#!/usr/bin/env node

/**
 * Test script to demonstrate the new orchestration functionality
 * Uses simulated embedding processes instead of real ones
 */

const chalk = require('chalk');
const { spawn } = require('child_process');
const path = require('path');

class TestWatchdogOrchestrator {
  constructor() {
    this.embeddingProgress = {
      kg: { active: false, progress: 0, status: 'idle' },
      regular: { active: false, progress: 0, status: 'idle' }
    };
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
    }, 500);
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
        path.join(__dirname, 'test_orchestration.py'),
        '--process', 'regular',
        '--duration', '8',
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
        const error = data.toString();
        if (!error.includes('WARNING')) {
          console.error(chalk.red(`Regular embeddings error: ${error}`));
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

  async triggerKgUpdateWithProgress() {
    return new Promise((resolve, reject) => {
      console.log(chalk.blue('   🔄 Starting KG auto-updater...'));
      
      this.embeddingProgress.kg.status = 'running';
      
      const kgProcess = spawn('python3', [
        path.join(__dirname, 'test_orchestration.py'),
        '--process', 'kg',
        '--duration', '12',
        '--progress-mode'
      ], {
        stdio: ['pipe', 'pipe', 'pipe'],
        cwd: path.join(__dirname, '..')
      });

      let outputBuffer = '';
      
      kgProcess.stdout.on('data', (data) => {
        outputBuffer += data.toString();
        this.parseEmbeddingProgress(outputBuffer, 'kg');
      });

      kgProcess.stderr.on('data', (data) => {
        const error = data.toString();
        if (!error.includes('WARNING')) {
          console.error(chalk.red(`KG updater error: ${error}`));
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

  parseEmbeddingProgress(output, type) {
    // Look for progress indicators in the output
    const progressMatch = output.match(/Progress:\s*(\d+(?:\.\d+)?)%/);
    if (progressMatch) {
      const progress = parseFloat(progressMatch[1]);
      this.embeddingProgress[type].progress = progress;
    }
    
    // Look for status updates
    const statusMatch = output.match(/Status:\s*([^\n]+)/);
    if (statusMatch) {
      this.embeddingProgress[type].status = statusMatch[1].trim();
    }
  }

  sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}

async function main() {
  console.log(chalk.cyan.bold('🧪 Testing Streamlined Watchdog Orchestration'));
  console.log(chalk.gray('━'.repeat(60)));

  const orchestrator = new TestWatchdogOrchestrator();

  console.log(chalk.yellow('📊 Simulating wiki changes detected...'));
  console.log(chalk.yellow('🔄 Triggering both embedding systems...'));

  const success = await orchestrator.triggerBothEmbeddingSystems();

  if (success) {
    console.log(chalk.green('\n✅ Orchestration test completed successfully!'));
    console.log(chalk.green('🔄 In real mode, watchdog would now resume monitoring...'));
  } else {
    console.log(chalk.red('\n❌ Orchestration test failed!'));
    console.log(chalk.yellow('⏳ In real mode, watchdog would wait before retry...'));
  }

  console.log(chalk.gray('\n━'.repeat(60)));
  console.log(chalk.cyan('🎉 Test demonstration complete!'));
}

if (require.main === module) {
  main().catch(console.error);
}
