import React, { Component } from "react";
import { ContentHeader } from '@components';
import { useNavigate } from "react-router-dom";

class HostView extends Component {
    constructor() {
        super();
        this.state = {
        };
    }

    render() {
        let props = this.props

        return (
            <h5>TODO</h5>
        )
    }
}

class Page extends Component {
    constructor() {
        super();
        this.state = {
            data: {},
        };
    }

    componentDidMount() {
    }

    componentWillUnmount() {
    }

    render() {
        let props = this.props
        return (
            <div>
                <ContentHeader title="Virtual Machines" />
                <section className="content">
                    <div className="container-fluid">
                        <div className="card">
                            <div className="card-header">
                                <div className="card-body">
                                    <HostView navigate={props.navigate} />
                                </div>
                            </div>
                        </div>
                    </div>
                </section>
            </div>
        );
    }
}

const PageWithNavigate = () => {
    const navigate = useNavigate();
    return (
        <Page navigate={navigate} />
    )
}

export default PageWithNavigate;
