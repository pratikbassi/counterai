import { useCallback, useState } from 'react';
import './DataCollectionBanner.css';

const STORAGE_KEY = 'counterai:data-collection-banner-dismissed';

function isDismissed(): boolean {
  if (typeof window === 'undefined') return false;
  try {
    return window.localStorage.getItem(STORAGE_KEY) === '1';
  } catch {
    return false;
  }
}

export default function DataCollectionBanner() {
  const [dismissed, setDismissed] = useState(isDismissed);

  const dismiss = useCallback(() => {
    try {
      window.localStorage.setItem(STORAGE_KEY, '1');
    } catch {
      /* ignore quota / private mode */
    }
    setDismissed(true);
  }, []);

  if (dismissed) {
    return null;
  }

  return (
    <div className="data-collection-banner" role="status">
      <p className="data-collection-banner__text">
        Files you upload are sent to our servers for analysis and may be stored or logged as part
        of normal service operation. Do not upload sensitive or personal data you are not permitted
        to share.
      </p>
      <button
        type="button"
        className="data-collection-banner__dismiss"
        onClick={dismiss}
        aria-label="Dismiss data collection notice"
      >
        Dismiss
      </button>
    </div>
  );
}
