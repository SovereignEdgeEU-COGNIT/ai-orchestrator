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
            vms: []
        };
    }

    async fetchData() {
        try {
            const response = await fetch('http://rocinante:8000/');
            if (!response.ok) {
                throw new Error('network response was not ok');
            }
            const rawPlacementLayout = await response.json();
            const placementLayout = rawPlacementLayout.sort((a, b) => Number(a.hostid) - Number(b.hostid));
            const vms = [];

            // Fetch VM details for each host
            const hostPromises = placementLayout.map(async host => {
                const vmPromises = host.vmids.map(async vmid => {
                    const vmResponse = await fetch(`http://rocinante:8000/vms/${vmid}`);
                    if (!vmResponse.ok) {
                        throw new Error(`Failed to fetch details for VM: ${vmid}`);
                    }
                    return await vmResponse.json();
                });
                const hostVMs = await Promise.all(vmPromises);
                vms.push(...hostVMs); // If you want to have a flat array of VMs, you can spread the hostVMs.
            });
            await Promise.all(hostPromises);

            return vms;
        } catch (error) {
            console.error("There was an error fetching the data:", error);
            return [];
        }
    }

    fetchVMDetails(vmid) {
        return fetch('http://rocinante:8000/vms/' + vmid)
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
        this.fetchData().then(vms => {
            this.setState({ vms });
        });

        this.interval = setInterval(() => {
            this.fetchData().then(vms => {
                this.setState({ vms });
            });
        }, 1000);
    }

    componentWillUnmount() {
        clearInterval(this.interval)
    }

    render() {
        let props = this.props

        const { vms } = this.state;
        const items = []
        if (vms == null) {
            return (<h5>No VM found</h5>)
        }

        let vmSorted = vms.sort((a, b) => {
            return Number(a.vm.vmid) - Number(b.vm.vmid);
        })

        for (let i = 0; i < vmSorted.length; i++) {
            let vm = vmSorted[i]
            items.push(<tr>
                <td> <i class="fas fa-cube"></i> &nbsp; {vm.vm.vmid}</td>
                <td> {vm.hostid}</td>
                <td> {vm.vm.cpu}</td>
                <td> {vm.vm.mem} bytes</td>
            </tr>)

        }

        return (
            <Table striped bordered hover >
                <thead>
                    <tr>
                        <th>VM Id</th>
                        <th>Host Id</th>
                        <th>CPU</th>
                        <th>Memory</th>
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
                <ContentHeader title="Virtual Machines" />
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
