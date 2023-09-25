import React, { useEffect, useState } from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { ToastContainer } from 'react-toastify';
import Main from '@modules/main/Main';
import { useWindowSize } from '@app/hooks/useWindowSize';
import { calculateWindowSize } from '@app/utils/helpers';
import { useDispatch, useSelector } from 'react-redux';
import { setWindowSize } from '@app/store/reducers/ui';
import Dashboard from '@pages/Dashboard';
import Hosts from '@pages/Hosts';
import VMS from '@pages/VMs';

const App = () => {
    const windowSize = useWindowSize();
    const screenSize = useSelector((state: any) => state.ui.screenSize);
    const dispatch = useDispatch();

    useEffect(() => {
        const size = calculateWindowSize(windowSize.width);
        if (screenSize !== size) {
            dispatch(setWindowSize(size));
        }
    }, [windowSize]);

    return (
        <BrowserRouter>
            <Routes>
                <Route path="/">
                    <Route path="/" element={<Main />}>
                        <Route path="/" element={<Dashboard />} />
                        <Route path="/hosts" element={<Hosts />} />
                        <Route path="/vms" element={<VMS />} />
                    </Route>
                </Route>
            </Routes>
            <ToastContainer
                autoClose={3000}
                draggable={false}
                position="top-right"
                hideProgressBar={false}
                newestOnTop
                closeOnClick
                rtl={false}
                pauseOnHover
            />
        </BrowserRouter>
    );
};

export default App;
