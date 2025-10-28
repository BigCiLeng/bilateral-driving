// 性能优化脚本
(function() {
  'use strict';

  // 视频懒加载和性能优化
  class VideoPerformanceManager {
    constructor() {
      this.loadedVideos = new Set();
      this.intersectionObserver = null;
      this.init();
    }

    init() {
      this.setupIntersectionObserver();
      this.setupVideoLazyLoading();
      this.optimizeVideoPlayback();
    }

    // 设置交叉观察器用于懒加载
    setupIntersectionObserver() {
      const options = {
        root: null,
        rootMargin: '50px',
        threshold: 0.1
      };

      this.intersectionObserver = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
          if (entry.isIntersecting) {
            this.loadVideo(entry.target);
            this.intersectionObserver.unobserve(entry.target);
          }
        });
      }, options);
    }

    // 设置视频懒加载
    setupVideoLazyLoading() {
      const videos = document.querySelectorAll('video[data-src]');
      videos.forEach(video => {
        this.intersectionObserver.observe(video);
      });
    }

    // 加载视频
    loadVideo(video) {
      if (this.loadedVideos.has(video.id)) return;

      const dataSrc = video.getAttribute('data-src');
      if (dataSrc) {
        // 显示加载状态
        this.showLoadingState(video);
        
        video.src = dataSrc;
        video.load();
        
        video.addEventListener('loadeddata', () => {
          this.hideLoadingState(video);
          this.loadedVideos.add(video.id);
        }, { once: true });

        video.addEventListener('error', () => {
          this.hideLoadingState(video);
          console.error('视频加载失败:', dataSrc);
        }, { once: true });
      }
    }

    // 显示加载状态
    showLoadingState(video) {
      const container = video.closest('.video-container');
      if (container) {
        const loading = container.querySelector('.video-loading');
        if (loading) {
          loading.style.display = 'flex';
        }
      }
    }

    // 隐藏加载状态
    hideLoadingState(video) {
      const container = video.closest('.video-container');
      if (container) {
        const loading = container.querySelector('.video-loading');
        if (loading) {
          loading.style.display = 'none';
        }
      }
    }

    // 优化视频播放
    optimizeVideoPlayback() {
      const videos = document.querySelectorAll('video');
      videos.forEach(video => {
        // 设置视频属性以优化性能
        video.setAttribute('playsinline', '');
        video.setAttribute('webkit-playsinline', '');
        
        // 添加性能优化样式
        video.style.willChange = 'transform';
        video.style.backfaceVisibility = 'hidden';
        video.style.transform = 'translateZ(0)';
        
        // 监听视频事件
        video.addEventListener('loadstart', () => {
          this.showLoadingState(video);
        });

        video.addEventListener('canplay', () => {
          this.hideLoadingState(video);
        });
      });
    }

    // 预加载指定组的视频
    preloadGroupVideos(groupId) {
      const group = document.getElementById(groupId);
      if (group) {
        const videos = group.querySelectorAll('video[data-src]');
        videos.forEach(video => {
          this.loadVideo(video);
        });
      }
    }
  }

  // 页面性能优化
  class PagePerformanceOptimizer {
    constructor() {
      this.init();
    }

    init() {
      this.optimizeImages();
      this.setupSmoothScrolling();
      this.optimizeAnimations();
      this.setupPreloading();
    }

    // 优化图片加载
    optimizeImages() {
      const images = document.querySelectorAll('img[data-src]');
      const imageObserver = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
          if (entry.isIntersecting) {
            const img = entry.target;
            img.src = img.dataset.src;
            img.classList.remove('lazy');
            imageObserver.unobserve(img);
          }
        });
      });

      images.forEach(img => imageObserver.observe(img));
    }

    // 设置平滑滚动
    setupSmoothScrolling() {
      document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
          e.preventDefault();
          const target = document.querySelector(this.getAttribute('href'));
          if (target) {
            target.scrollIntoView({
              behavior: 'smooth',
              block: 'start'
            });
          }
        });
      });
    }

    // 优化动画性能
    optimizeAnimations() {
      // 使用 requestAnimationFrame 优化动画
      const animateElements = document.querySelectorAll('.fade-in');
      const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
          if (entry.isIntersecting) {
            requestAnimationFrame(() => {
              entry.target.classList.add('visible');
            });
          }
        });
      }, {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
      });

      animateElements.forEach(el => observer.observe(el));
    }

    // 设置预加载
    setupPreloading() {
      // 预加载关键资源
      const criticalResources = [
        'static/videos/teaser.mp4',
        'static/images/paradigm.png',
        'static/images/framework.png'
      ];

      criticalResources.forEach(resource => {
        const link = document.createElement('link');
        link.rel = 'preload';
        link.href = resource;
        link.as = resource.endsWith('.mp4') ? 'video' : 'image';
        document.head.appendChild(link);
      });
    }
  }

  // 内存管理
  class MemoryManager {
    constructor() {
      this.cleanupTasks = [];
      this.init();
    }

    init() {
      // 页面卸载时清理资源
      window.addEventListener('beforeunload', () => {
        this.cleanup();
      });

      // 定期清理不需要的资源
      setInterval(() => {
        this.cleanup();
      }, 30000); // 每30秒清理一次
    }

    addCleanupTask(task) {
      this.cleanupTasks.push(task);
    }

    cleanup() {
      this.cleanupTasks.forEach(task => {
        try {
          task();
        } catch (error) {
          console.error('清理任务执行失败:', error);
        }
      });
    }
  }

  // 初始化所有优化器
  document.addEventListener('DOMContentLoaded', function() {
    const videoManager = new VideoPerformanceManager();
    const pageOptimizer = new PagePerformanceOptimizer();
    const memoryManager = new MemoryManager();

    // 将视频管理器暴露到全局作用域
    window.videoManager = videoManager;

    // 预加载关键演示组，确保交互视频可用
    videoManager.preloadGroupVideos('challenge-demos');

    // 添加清理任务
    memoryManager.addCleanupTask(() => {
      // 清理未使用的视频元素
      const videos = document.querySelectorAll('video');
      videos.forEach(video => {
        const isInteractive = video.hasAttribute('controls');
        if (!isInteractive && video.paused && video.readyState >= 2) {
          video.src = '';
          video.load();
        }
      });
    });

    console.log('性能优化模块已加载');
  });

})();
