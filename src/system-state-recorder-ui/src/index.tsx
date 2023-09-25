
import "@fontsource/titillium-web";
import { createRoot } from 'react-dom/client';
import { Provider } from 'react-redux';
import App from '@app/App';
import store from '@store/store';
import './utils/i18n';
import './index.css';
import * as serviceWorker from './serviceWorker';
import "@fontsource/poppins"; // Defaults to weight 400
import "@fontsource/poppins/400.css"; // Specify weight
import "@fontsource/poppins/400-italic.css"; // Specify weight and style

declare const window: any;

window.PF = {
    config: {
        mode: 'bs4'
    }
};

const container: any = document.getElementById('root');
const root = createRoot(container);
root.render(
    <Provider store={store}>
        <App />
    </Provider>
);

serviceWorker.unregister();
