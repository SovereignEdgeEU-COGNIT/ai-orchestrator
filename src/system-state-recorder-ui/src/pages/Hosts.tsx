import React, { Component } from "react";
import { ContentHeader } from '@components';
import { useNavigate } from "react-router-dom";
import Table from 'react-bootstrap/Table';
import { raw } from "express";
import Button from 'react-bootstrap/Button';

class HostsView extends Component {
    constructor() {
        super();
        this.state = {
            hosts: []
        };
    }

    fetchData() {
        let hosts = [];
        return fetch('http://rocinante:8000/')
            .then(this.handleErrors)
            .then(response => response.json())
            .then(rawPlacementLayout => {
                let placementLayout = rawPlacementLayout.sort((a, b) => {
                    return Number(a.hostid) - Number(b.hostid);
                });

                let fetchPromises = placementLayout.map(host => {
                    return this.fetchHostDetails(host.hostid)
                        .then(h => {
                            h.id = host.hostid;
                            hosts.push(h);
                        });
                });

                return Promise.all(fetchPromises).then(() => hosts);
            });
    }

    fetchHostDetails(hostid) {
        return fetch('http://rocinante:8000/hosts/' + hostid)
            .then(this.handleErrors)
            .then(response => response.json());
    }

    handleErrors(response) {
        if (!response.ok) {
            throw new Error('network response was not ok');
        }
        return response;
    }

    componentDidMount() {

        this.fetchData().then(hosts => {
            this.setState({ hosts });
        });

        this.interval = setInterval(() => {
            this.fetchData().then(hosts => {
                this.setState({ hosts });
            });
        }, 1000);
    }

    componentWillUnmount() {
        clearInterval(this.interval)
    }

    render() {
        let props = this.props

        const setRenewable = (e, hostid, renewableEnergy) => {
            fetch("http://rocinante:8000/set?hostid=" + hostid + "&renewable=" + renewableEnergy.toString(), {
                method: 'GET',
            })
                .then(response => {
                    if (!response.ok) {
                        throw new Error('network response was not ok');
                    }
                    return response.json(); // or response.text() if it's plain text
                })
                .then(data => {
                    console.log(data);
                })
                .catch(error => {
                    console.error('there was a problem with the fetch operation:', error);
                });
        }

        const { hosts } = this.state;
        const items = []
        if (hosts == null) {
            return (<h5>No hosts found</h5>)
        }

        let hostsSorted = hosts.sort((a, b) => {
            return Number(a.id) - Number(b.id);
        })

        for (let i = 0; i < hostsSorted.length; i++) {
            let host = hosts[i]
            if (host.state.renewable_energy) {
                items.push(<tr>
                    <td> <i class="fas fa-server"></i> &nbsp; {host.id}</td>
                    <td> {host.state.renewable_energy.toString()}</td>
                    <td> {host.cpu_total}</td>
                    <td> {host.total_mem_bytes} bytes</td>
                    <td> {host.cpu_total - host.cpu_usage}</td>
                    <td> {host.total_mem_bytes - host.usage_mem_bytes} bytes</td>
                    <td> {host.vms}</td>
                    <td>
                        <Button variant="secondary" onClick={(e) => setRenewable(e, host.id, false)}>
                            Unset
                        </Button>
                    </td>
                </tr>)
            } else {
                items.push(<tr>
                    <td> <i class="fas fa-server"></i> &nbsp; {host.id}</td>
                    <td> {host.state.renewable_energy.toString()}</td>
                    <td> {host.cpu_total}</td>
                    <td> {host.total_mem_bytes} bytes</td>
                    <td> {host.cpu_total - host.cpu_usage}</td>
                    <td> {host.total_mem_bytes - host.usage_mem_bytes} bytes</td>
                    <td> {host.vms}</td>
                    <td>
                        <Button variant="secondary" onClick={(e) => setRenewable(e, host.id, true)}>
                            Set
                        </Button>
                    </td>
                </tr>)
            }

        }

        return (
            <Table striped bordered hover >
                <thead>
                    <tr>
                        <th>Host Id</th>
                        <th>Renewable</th>
                        <th>Total CPU</th>
                        <th>Total Memory</th>
                        <th>Available CPU:</th>
                        <th>Available Memory:</th>
                        <th>VMs:</th>
                        <th>Renewable Energy:</th>
                    </tr>
                </thead>
                <tbody>
                    {items}
                </tbody>
            </Table >
        );
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
                <ContentHeader title="Hosts" />
                <section className="content">
                    <div className="container-fluid">
                        <div className="card">
                            <div className="card-header">
                                <div className="card-body">
                                    <HostsView navigate={props.navigate} />
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
