(() => {
    document.addEventListener("DOMContentLoaded", () => {
    const composerForm = document.querySelector("form.composer");
    const textarea = composerForm?.querySelector("#manual-message");
    if (!composerForm || !textarea) return;

      textarea.addEventListener("keydown", (ev) => {
        if (ev.key === "Enter" && !ev.shiftKey) {
          ev.preventDefault();
          composerForm.requestSubmit(); 
        }
      });
    });

  function initHistoryDashboard() {
    const container = document.querySelector('[data-history-mode="dashboard"]');
    if (!container) {
      return;
    }

    const resizer = container.querySelector('[data-sidebar-resizer]');
    const contactStore = container.querySelector('.contact-store');
    const contactStoreResizer = container.querySelector('[data-contact-store-resizer]');
    initSidebarResizer();
    initContactStoreResizer();

    const chatWindow = container.querySelector('.chat-window');
    const messageCountEl = container.querySelector('.chat-header .sub');
    const feedUrl = container.dataset.feedUrl || '';
    const initialMessages = parseDatasetJson(container.dataset.initialMessages);
    const initialState = parseDatasetJson(container.dataset.initialState);
    const pollIntervalMs = Number(container.dataset.pollIntervalMs || 5000);

    let lastMessageSignature = computeSignature(initialMessages);
    let lastStateSignature = initialState
      ? JSON.stringify([
          !!initialState.ai_paused,
          initialState.handoff_reason || '',
          initialState.handoff_ts || '',
        ])
      : null;
    let isFetching = false;

    scrollToEnd();

    if (!feedUrl) {
      return;
    }

    fetchUpdates();
    setInterval(fetchUpdates, pollIntervalMs);

    function initSidebarResizer() {
      if (!resizer) {
        return;
      }

      const storageKey = 'history.sidebarWidth';
      let minWidth = 220;
      let maxWidth = 520;
      let lastWidth = null;
      let isDragging = false;

      const parseSize = (value) => {
        if (value == null) {
          return null;
        }
        const numeric = String(value).trim();
        if (!numeric) {
          return null;
        }
        const parsed = parseInt(numeric.replace(/[^0-9.]/g, ''), 10);
        return Number.isFinite(parsed) ? parsed : null;
      };

      const clampWidth = (value) => Math.min(Math.max(value, minWidth), maxWidth);

      const refreshBounds = () => {
        const styles = window.getComputedStyle(container);
        const min = parseSize(styles.getPropertyValue('--sidebar-min-width'));
        const max = parseSize(styles.getPropertyValue('--sidebar-max-width'));
        if (min !== null) {
          minWidth = min;
        }
        if (max !== null) {
          maxWidth = max;
        }
        if (maxWidth < minWidth) {
          maxWidth = minWidth;
        }
        resizer.setAttribute('aria-valuemin', String(minWidth));
        resizer.setAttribute('aria-valuemax', String(maxWidth));
      };

      const persistWidth = (value) => {
        try {
          window.localStorage.setItem(storageKey, String(value));
        } catch (error) {
          console.warn('Failed to persist sidebar width:', error);
        }
      };

      const applyWidth = (value) => {
        const width = clampWidth(value);
        lastWidth = width;
        container.style.setProperty('--sidebar-width', `${width}px`);
        resizer.setAttribute('aria-valuenow', String(width));
        return width;
      };

      const getCurrentWidth = () => {
        if (lastWidth !== null) {
          return lastWidth;
        }
        const styles = window.getComputedStyle(container);
        const current = parseSize(styles.getPropertyValue('--sidebar-width'));
        return current !== null ? clampWidth(current) : clampWidth(320);
      };

      const updateWidthFromPointer = (clientX) => {
        const rect = container.getBoundingClientRect();
        applyWidth(clientX - rect.left);
      };

      const handlePointerMove = (event) => {
        if (!isDragging) {
          return;
        }
        const point = event.touches ? event.touches[0] : event;
        if (!point || typeof point.clientX !== 'number') {
          return;
        }
        event.preventDefault();
        updateWidthFromPointer(point.clientX);
      };

      const stopDragging = () => {
        if (!isDragging) {
          return;
        }
        isDragging = false;
        document.body.style.removeProperty('user-select');
        resizer.classList.remove('is-dragging');
        window.removeEventListener('mousemove', handlePointerMove);
        window.removeEventListener('touchmove', handlePointerMove);
        window.removeEventListener('mouseup', stopDragging);
        window.removeEventListener('touchend', stopDragging);
        window.removeEventListener('touchcancel', stopDragging);
        if (lastWidth !== null) {
          persistWidth(lastWidth);
        }
      };

      const startDragging = (event) => {
        const point = event.touches ? event.touches[0] : event;
        if (!point || typeof point.clientX !== 'number') {
          return;
        }
        event.preventDefault();
        refreshBounds();
        isDragging = true;
        resizer.classList.add('is-dragging');
        document.body.style.userSelect = 'none';
        updateWidthFromPointer(point.clientX);
        window.addEventListener('mousemove', handlePointerMove, { passive: false });
        window.addEventListener('touchmove', handlePointerMove, { passive: false });
        window.addEventListener('mouseup', stopDragging);
        window.addEventListener('touchend', stopDragging);
        window.addEventListener('touchcancel', stopDragging);
      };

      const defaultWidth = (() => {
        const styles = window.getComputedStyle(container);
        const value = parseSize(styles.getPropertyValue('--sidebar-width'));
        return value !== null ? clampWidth(value) : clampWidth(320);
      })();

      refreshBounds();
      const storedWidth = parseSize(window.localStorage.getItem(storageKey));
      if (storedWidth !== null) {
        applyWidth(storedWidth);
      } else {
        applyWidth(defaultWidth);
      }

      resizer.addEventListener('mousedown', startDragging);
      resizer.addEventListener('touchstart', startDragging, { passive: false });

      resizer.addEventListener('keydown', (event) => {
        let delta = 0;
        switch (event.key) {
          case 'ArrowLeft':
          case 'Left':
            delta = -10;
            break;
          case 'ArrowRight':
          case 'Right':
            delta = 10;
            break;
          case 'Home':
            event.preventDefault();
            refreshBounds();
            applyWidth(minWidth);
            persistWidth(minWidth);
            return;
          case 'End':
            event.preventDefault();
            refreshBounds();
            applyWidth(maxWidth);
            persistWidth(maxWidth);
            return;
          default:
            return;
        }
        event.preventDefault();
        refreshBounds();
        const nextWidth = clampWidth(getCurrentWidth() + delta);
        applyWidth(nextWidth);
        persistWidth(nextWidth);
      });

      resizer.addEventListener('dblclick', () => {
        refreshBounds();
        applyWidth(defaultWidth);
        persistWidth(defaultWidth);
      });

      window.addEventListener('resize', () => {
        refreshBounds();
        if (lastWidth !== null) {
          applyWidth(lastWidth);
        }
      });
    }

    function initContactStoreResizer() {
      if (!contactStore || !contactStoreResizer) {
        return;
      }

      const sidebar = container.querySelector('.sidebar');
      if (!sidebar) {
        return;
      }

      const storageKey = 'history.contactStoreHeight';
      let minHeight = 160;
      let maxHeight = 520;
      let lastHeight = null;
      let isDragging = false;

      const parseSize = (value) => {
        if (value == null) {
          return null;
        }
        const numeric = String(value).trim();
        if (!numeric) {
          return null;
        }
        const parsed = parseInt(numeric.replace(/[^0-9.]/g, ''), 10);
        return Number.isFinite(parsed) ? parsed : null;
      };

      const clampHeight = (value) => Math.min(Math.max(value, minHeight), maxHeight);

      const refreshBounds = () => {
        const styles = window.getComputedStyle(sidebar);
        const parsedMin = parseSize(styles.getPropertyValue('--contact-store-min-height'));
        if (parsedMin !== null) {
          minHeight = parsedMin;
        }
        const sidebarRect = sidebar.getBoundingClientRect();
        const header = sidebar.querySelector('header');
        const headerHeight = header ? header.getBoundingClientRect().height : 0;
        const minContactListHeight = parseSize(styles.getPropertyValue('--contact-list-min-height')) || 140;
        const available = sidebarRect.height - headerHeight - minContactListHeight;
        maxHeight = Math.max(minHeight, available);
        contactStoreResizer.setAttribute('aria-valuemin', String(minHeight));
        contactStoreResizer.setAttribute('aria-valuemax', String(Math.round(maxHeight)));
      };

      const persistHeight = (value) => {
        try {
          window.localStorage.setItem(storageKey, String(value));
        } catch (error) {
          console.warn('Failed to persist contacts panel height:', error);
        }
      };

      const applyHeight = (value) => {
        const height = clampHeight(value);
        lastHeight = height;
        sidebar.style.setProperty('--contact-store-height', `${height}px`);
        contactStoreResizer.setAttribute('aria-valuenow', String(Math.round(height)));
        return height;
      };

      const getDefaultHeight = () => {
        const styles = window.getComputedStyle(sidebar);
        const raw = parseSize(styles.getPropertyValue('--contact-store-height'));
        return raw !== null ? clampHeight(raw) : clampHeight(260);
      };

      const updateHeightFromPointer = (clientY) => {
        const sidebarRect = sidebar.getBoundingClientRect();
        applyHeight(sidebarRect.bottom - clientY);
      };

      const handlePointerMove = (event) => {
        if (!isDragging) {
          return;
        }
        const point = event.touches ? event.touches[0] : event;
        if (!point || typeof point.clientY !== 'number') {
          return;
        }
        event.preventDefault();
        updateHeightFromPointer(point.clientY);
      };

      const stopDragging = () => {
        if (!isDragging) {
          return;
        }
        isDragging = false;
        document.body.style.removeProperty('user-select');
        contactStoreResizer.classList.remove('is-dragging');
        window.removeEventListener('mousemove', handlePointerMove);
        window.removeEventListener('touchmove', handlePointerMove);
        window.removeEventListener('mouseup', stopDragging);
        window.removeEventListener('touchend', stopDragging);
        window.removeEventListener('touchcancel', stopDragging);
        if (lastHeight !== null) {
          persistHeight(lastHeight);
        }
      };

      const startDragging = (event) => {
        const point = event.touches ? event.touches[0] : event;
        if (!point || typeof point.clientY !== 'number') {
          return;
        }
        event.preventDefault();
        refreshBounds();
        isDragging = true;
        contactStoreResizer.classList.add('is-dragging');
        document.body.style.userSelect = 'none';
        updateHeightFromPointer(point.clientY);
        window.addEventListener('mousemove', handlePointerMove, { passive: false });
        window.addEventListener('touchmove', handlePointerMove, { passive: false });
        window.addEventListener('mouseup', stopDragging);
        window.addEventListener('touchend', stopDragging);
        window.addEventListener('touchcancel', stopDragging);
      };

      contactStoreResizer.addEventListener('mousedown', startDragging);
      contactStoreResizer.addEventListener('touchstart', startDragging, { passive: false });

      contactStoreResizer.addEventListener('keydown', (event) => {
        let delta = 0;
        switch (event.key) {
          case 'ArrowUp':
          case 'Up':
            delta = 12;
            break;
          case 'ArrowDown':
          case 'Down':
            delta = -12;
            break;
          case 'Home':
            event.preventDefault();
            refreshBounds();
            applyHeight(minHeight);
            persistHeight(minHeight);
            return;
          case 'End':
            event.preventDefault();
            refreshBounds();
            applyHeight(maxHeight);
            persistHeight(maxHeight);
            return;
          default:
            return;
        }
        event.preventDefault();
        refreshBounds();
        const nextHeight = clampHeight((lastHeight !== null ? lastHeight : getDefaultHeight()) + delta);
        applyHeight(nextHeight);
        persistHeight(nextHeight);
      });

      contactStoreResizer.addEventListener('dblclick', () => {
        refreshBounds();
        const defaultHeight = getDefaultHeight();
        applyHeight(defaultHeight);
        persistHeight(defaultHeight);
      });

      refreshBounds();
      const storedHeight = parseSize(window.localStorage.getItem(storageKey));
      if (storedHeight !== null) {
        applyHeight(storedHeight);
      } else {
        applyHeight(getDefaultHeight());
      }

      window.addEventListener('resize', () => {
        refreshBounds();
        if (lastHeight !== null) {
          applyHeight(lastHeight);
        }
      });
    }

    function parseDatasetJson(raw) {
      if (!raw) {
        return null;
      }
      try {
        return JSON.parse(raw);
      } catch (error) {
        console.warn('Failed to parse dashboard payload:', error);
        return null;
      }
    }

    function computeSignature(messages) {
      if (!Array.isArray(messages) || messages.length === 0) {
        return '0|';
      }
      const last = messages[messages.length - 1] || {};
      return `${messages.length}|${last.timestamp || ''}|${last.role || ''}|${
        (last.content || '').length
      }`;
    }

    function scrollToEnd() {
      if (chatWindow) {
        chatWindow.scrollTop = chatWindow.scrollHeight;
      }
    }

    function updateMessageCount(count) {
      if (!messageCountEl) {
        return;
      }
      const suffix = count === 1 ? '' : 's';
      messageCountEl.textContent = `${count} message${suffix}`;
    }

    function renderMessageContent(containerEl, text) {
      containerEl.textContent = '';
      const lines = String(text || '').split(/\r?\n/);
      lines.forEach((line, index) => {
        if (index > 0) {
          containerEl.appendChild(document.createElement('br'));
        }
        containerEl.appendChild(document.createTextNode(line));
      });
    }

    function renderMessages(messages) {
      if (!chatWindow) {
        return;
      }
      chatWindow.innerHTML = '';
      (messages || []).forEach((message) => {
        const role = ((message && message.role) || 'unknown').toLowerCase();
        let rowClass = 'system';
        let bubbleClass = 'system';
        if (role === 'assistant') {
          rowClass = 'assistant';
          bubbleClass = 'assistant';
        } else if (role === 'user') {
          rowClass = 'user';
          bubbleClass = 'user';
        } else if (role === 'operator') {
          rowClass = 'assistant';
          bubbleClass = 'operator';
        }

        const row = document.createElement('div');
        row.className = `bubble-row ${rowClass}`;

        const bubble = document.createElement('div');
        bubble.className = `bubble ${bubbleClass}`;
        if (message && message.ai_readable === false) {
          bubble.classList.add('muted');
        }

        const roleChip = document.createElement('div');
        roleChip.className = 'role-chip';
        roleChip.textContent = message && message.role ? message.role : 'Unknown';
        bubble.appendChild(roleChip);

        const bubbleContent = document.createElement('div');
        bubbleContent.className = 'bubble-content';
        renderMessageContent(bubbleContent, message && message.content ? message.content : '');
        bubble.appendChild(bubbleContent);

        if (message && message.ai_readable === false) {
          const metaTag = document.createElement('div');
          metaTag.className = 'meta-tag';
          metaTag.textContent = 'Not forwarded to AI';
          bubble.appendChild(metaTag);
        }

        if (message && message.timestamp) {
          const timestamp = document.createElement('div');
          timestamp.className = 'timestamp';
          timestamp.textContent = message.timestamp;
          bubble.appendChild(timestamp);
        }

        row.appendChild(bubble);
        chatWindow.appendChild(row);
      });
      updateMessageCount(Array.isArray(messages) ? messages.length : 0);
      scrollToEnd();
    }

    function fetchUpdates() {
      if (isFetching) {
        return;
      }
      isFetching = true;
      fetch(feedUrl, { cache: 'no-store' })
        .then((response) => {
          if (!response.ok) {
            throw new Error(`Failed to poll feed (${response.status})`);
          }
          return response.json();
        })
        .then((data) => {
          if (!data || !Array.isArray(data.messages)) {
            return;
          }
          const messageSignature = computeSignature(data.messages);
          if (messageSignature !== lastMessageSignature) {
            renderMessages(data.messages);
            lastMessageSignature = messageSignature;
          }
          const stateSignature = JSON.stringify([
            data.ai_paused,
            data.handoff_reason || '',
            data.handoff_ts || '',
          ]);
          if (lastStateSignature === null) {
            lastStateSignature = stateSignature;
          } else if (stateSignature !== lastStateSignature) {
            lastStateSignature = stateSignature;
            window.location.reload();
          }
        })
        .catch((error) => {
          console.error('Failed to poll conversation feed:', error);
        })
        .finally(() => {
          isFetching = false;
        });
    }

  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initHistoryDashboard);
  } else {
    initHistoryDashboard();
  }
})();
