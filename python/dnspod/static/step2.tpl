<p>选择域名：
<select class="selDomains">
    {{#domains}}
    <option value={{id}}>{{name}}</option>
    {{/domains}}
</select>
</p>
<table class="table">
    <thead>
        <tr>
            <th>#</th>
            <th>主机</th>
            <th>类型</th>
            <th>线路</th>
            <th>记录值</th>
            <th>MX优先级</th>
            <th>TTL</th>
            <th>状态</th>
        </tr>
    </thead>
    <tbody>
        {{#records}}
        <tr>
            <td>{{seq}}</td>
            <td>{{sub_domain}}</td>
            <td>{{record_type}}</td>
            <td>{{record_line}}</td>
            <td>{{value}}</td>
            <td>{{mx}}</td>
            <td>{{ttl}}</td>
            <td id="record_status_{{seq}}">{{state}}</td>
        </tr>
        {{/records}}
    </tbody>
</table>
<button class="back">上一步</button>
<button class="import_record">导入</button>
