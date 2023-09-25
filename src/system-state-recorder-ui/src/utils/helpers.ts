import { argv0 } from "process";

export const sleep = (time: number) =>
    new Promise((res) => setTimeout(res, time));

export const calculateWindowSize = (windowWidth: number) => {
    if (windowWidth >= 1200) {
        return 'lg';
    }
    if (windowWidth >= 992) {
        return 'md';
    }
    if (windowWidth >= 768) {
        return 'sm';
    }
    return 'xs';
};

export const setWindowClass = (classList: string) => {
    const window: HTMLElement | null =
        document && document.getElementById('root');
    if (window) {
        // @ts-ignore
        window.classList = classList;
    }
};
export const addWindowClass = (classList: string) => {
    const window: HTMLElement | null =
        document && document.getElementById('root');
    if (window) {
        // @ts-ignore
        window.classList.add(classList);
    }
};

export const removeWindowClass = (classList: string) => {
    const window: HTMLElement | null =
        document && document.getElementById('root');
    if (window) {
        // @ts-ignore
        window.classList.remove(classList);
    }
};

export const bool2str = (b) => {
    if (b) {
        return "true"
    }
    return "false"
};

export const attrtype2str = (attrtype) => {
    if (attrtype == 0) {
        return "In"
    } else if (attrtype == 1) {
        return "Out"
    } else if (attrtype == 2) {
        return "Err"
    } else if (attrtype == 3) {
        return "Env"
    } else {
        return "Unknown"
    }
}

export const state2str = (state) => {
    if (state == 0) {
        return "Waiting"
    } else if (state == 1) {
        return "Running"
    } else if (state == 2) {
        return "Successful"
    } else if (state == 3) {
        return "Failed"
    } else {
        return "Unknown"
    }
};

export const rtstate2str = (state) => {
    if (state == 0) {
        return "Pending"
    } else if (state == 1) {
        return "Approved"
    } else if (state == 2) {
        return "Rejected"
    } else {
        return "Unknown"
    }
};

export const parseTime = (time) => {
    let unixTimestamp = Date.parse(time)
    if (unixTimestamp < 0) {
        return "n/a"
    }

    let date = new Date(unixTimestamp);
    var year = date.getFullYear().toString();
    var month = date.toLocaleString("default", { month: "2-digit" })
    var day = date.toLocaleString("default", { day: "2-digit" })

    let hours = date.getHours()
    if (hours < 10) {
        hours = "0" + hours
    }

    let minutes = date.getMinutes()
    if (minutes < 10) {
        minutes = "0" + minutes
    }

    let seconds = date.getSeconds()
    if (seconds < 10) {
        seconds = "0" + seconds
    }
    return year + "-" + month + "-" + day + " " + hours + ":" + minutes + ":" + seconds;
};

export const calcWaitTime = (state, submissiontime, starttime, endtime) => {
    if (state == 0) { // Waiting 
        return (Date.now() - Date.parse(submissiontime)) / 1000
    } else if (state == 1 || state == 2 || state == 3) { // Successful or Failed 
        let s = Date.parse(starttime)
        if (s > 0) {
            return (Date.parse(starttime) - Date.parse(submissiontime)) / 1000
        } else {
            return (Date.parse(endtime) - Date.parse(submissiontime)) / 1000
        }
    }
}

export const calcExecTime = (state, starttime, endtime) => {
    if (state == 0) { // Waiting 
        return 0
    } else if (state == 1) { // Running 
        return (Date.now() - Date.parse(starttime)) / 1000
    } else if (state == 2 || state == 3) { // Successful or Failed 
        let s = Date.parse(starttime)
        if (s > 0) {
            return (Date.parse(endtime) - s) / 1000
        } else {
            return 0
        }
    }
}

export const calcRemainingTime = (currentState, state, deadline) => {
    let d = Date.parse(deadline)
    if (currentState == state && d > 0) { // Running 
        return (d - Date.now()) / 1000
    } else {
        return "n/a"
    }
}

export const parseDict = (dict) => {
    let str = ""
    for (const key in dict) {
        str += key + ":" + dict[key] + " "
    }

    return str
}

export const parseArr = (arr) => {
    let str = ""
    for (const i in arr) {
        str += arr[i] + ", "
    }

    if (str.length > 1) {
        str = str.slice(0, -2);
    }

    return str
}
